#include "anira_tilde/inference/Session.h"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <stdexcept>


namespace anira_tilde {

static anira::ContextConfig load_context_config(anira::JsonConfigLoader& loader) {
    auto ptr = loader.get_context_config();
    if (!ptr) throw std::runtime_error("Failed to load context config (JSON may be malformed)");
    return std::move(*ptr);
}

static anira::InferenceConfig load_inference_config(anira::JsonConfigLoader& loader, const std::string& path) {
    auto ptr = loader.get_inference_config();
    if (!ptr) throw std::runtime_error("Failed to load inference config from: " + path + " (JSON may be malformed or missing 'inference_config' key)");
    auto config = std::move(*ptr);

    auto config_dir = std::filesystem::path(path).parent_path();
    for (auto& md : config.m_model_data) {
        if (!md.m_is_binary) {
            auto model_path = std::filesystem::path(std::string(static_cast<char*>(md.m_data), md.m_size));
            if (model_path.is_relative()) {
                config.set_model_path((config_dir / model_path).string(), md.m_backend);
            }
        }
    }

    // State passing requires strictly sequential inference: if two workers race,
    // pre_process(N+1) may read stale state before post_process(N) has written it.
    if (!parse_state_pairs(path).empty()) {
        config.m_num_parallel_processors = 1;
    }
    return config;
}

Session::Session(std::string json_config_path) :
    m_config_loader(json_config_path),
    m_anira_context(load_context_config(m_config_loader)),
    m_inference_config(load_inference_config(m_config_loader, json_config_path)),
    m_state_pairs(parse_state_pairs(json_config_path)),
    m_pp_processor(m_inference_config, m_state_pairs),
    m_inference_handler(m_pp_processor, m_inference_config),
    m_layout(TensorLayout::from(m_inference_config, m_state_pairs))
{}

namespace {

/// For the first upsample tensor (input_size < output_size), compute the
/// extra "decoder" latency anira needs to allocate. Returns std::nullopt
/// when no tensor needs it.
struct DecoderLatency { size_t tensor_index; unsigned int samples; };

std::optional<DecoderLatency> compute_decoder_latency(const TensorLayout& layout,
                                                      size_t buffer_size) {
    for (size_t i = 0; i < layout.output_block_sizes.size(); ++i) {
        if (i >= layout.input_block_sizes.size()) break;
        if (layout.input_block_sizes[i] >= layout.output_block_sizes[i]) continue;
        const size_t out_sz        = layout.output_block_sizes[i];
        const size_t safety_margin = std::max(buffer_size, out_sz);
        return DecoderLatency{ i, static_cast<unsigned int>(out_sz + safety_margin) };
    }
    return std::nullopt;
}

} // namespace

void Session::prepare(size_t buffer_size, double sample_rate) {
    anira::HostConfig host_config {
        static_cast<float>(buffer_size),
        static_cast<float>(sample_rate),
    };

    if (const auto dl = compute_decoder_latency(m_layout, buffer_size)) {
        m_inference_handler.prepare(host_config, dl->samples, dl->tensor_index);
    } else {
        m_inference_handler.prepare(host_config);
    }

    m_selected_backend = anira::InferenceBackend::LIBTORCH;
    m_inference_handler.set_inference_backend(m_selected_backend);

    m_rate_adaptor.prepare(m_layout, buffer_size);
}

size_t Session::get_latency_samples()
{
    return m_inference_handler.get_latency();
}

void Session::process(const float* const* const* input_data,  size_t* num_input_samples,
                      float* const* const*       output_data, size_t* num_output_samples)
{
    const auto v = m_rate_adaptor.pre_dispatch(m_layout,
                                               input_data,  num_input_samples,
                                               output_data, num_output_samples);

    // State-passing requires pop-before-push so post_process(N-1) writes
    // fresh state before pre_process(N) reads it.
    if (!m_state_pairs.empty()) {
        m_inference_handler.pop_data (v.out_tensors, v.out_sample_counts);
        m_inference_handler.push_data(v.in_tensors,  v.in_sample_counts);
    } else {
        m_inference_handler.push_data(v.in_tensors,  v.in_sample_counts);
        m_inference_handler.pop_data (v.out_tensors, v.out_sample_counts);
    }

    m_rate_adaptor.post_dispatch(m_layout, output_data, num_output_samples);
}

void Session::set_input(float input, size_t tensor_index, size_t channel) {
    m_pp_processor.set_input(input, tensor_index, channel);
}

float Session::get_output(size_t tensor_index, size_t channel) {
    return m_pp_processor.get_output(tensor_index, channel);
}

} // namespace anira_tilde
