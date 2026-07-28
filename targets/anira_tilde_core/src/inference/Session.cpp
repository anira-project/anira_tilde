#include "anira_tilde/inference/Session.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "anira/ContextConfig.h"
#include "anira/InferenceConfig.h"
#include "anira/utils/HostConfig.h"
#include "anira/utils/InferenceBackend.h"
#include "anira/utils/JsonConfigLoader.h"
#include "anira_tilde/inference/TensorLayout.h"
#include "anira_tilde/resampling/ResamplerConfigParser.h"
#include "anira_tilde/state_passing/StatePairParser.h"

#ifdef USE_LIBTORCH
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include <torch/script.h>
#pragma GCC diagnostic pop
#endif

namespace anira_tilde {

namespace {

anira::ContextConfig load_context_config(anira::JsonConfigLoader& loader) {
    auto ptr = loader.get_context_config();
    if (!ptr) { throw std::runtime_error("Failed to load context config (JSON may be malformed)"); }
    return std::move(*ptr);
}

anira::InferenceConfig load_inference_config(anira::JsonConfigLoader& loader,
                                             const std::string& path) {
    auto ptr = loader.get_inference_config();
    if (!ptr) {
        throw std::runtime_error("Failed to load inference config from: " + path +
                                 " (JSON may be malformed or missing 'inference_config' key)");
    }
    auto config = std::move(*ptr);

    auto config_dir = std::filesystem::path(path).parent_path();
    for (auto& md : config.m_model_data) {
        if (!md.m_is_binary) {
            auto model_path =
                std::filesystem::path(std::string(static_cast<char*>(md.m_data), md.m_size));
            if (model_path.is_relative()) {
                config.set_model_path((config_dir / model_path).string(), md.m_backend);
            }
        }
    }

#ifdef USE_LIBTORCH
    // Validate any model_function names and load LibTorch models before threads
    // start - get_method throws a fatal LibTorch assertion if the name doesn't
    // exist in the model.
    for (auto& md : config.m_model_data) {
        if (md.m_backend != anira::InferenceBackend::LIBTORCH || md.m_is_binary) { continue; }
        const std::string model_path(static_cast<char*>(md.m_data), md.m_size);
        try {
            auto module = torch::jit::load(model_path);
            if (!md.m_model_function.empty() && !module.find_method(md.m_model_function)) {
                throw std::runtime_error("model '" + model_path + "' has no method '" +
                                         md.m_model_function +
                                         "' - check model function in your JSON config");
            }
        } catch (const c10::Error& e) {
            throw std::runtime_error("failed to load model '" + model_path + "': " + e.what());
        }
    }
#endif

    // State passing requires strictly sequential inference: if two workers race,
    // pre_process(N+1) may read stale state before post_process(N) has written it.
    if (!parse_state_pairs(path).empty()) { config.m_num_parallel_processors = 1; }

    // Force a per-session processor. anira's shared processor pool stores the
    // InferenceConfig *by reference* (BackendBase::m_inference_config), bound to
    // whichever session first created the processor. If two instances share a
    // config and the first is closed, that reference dangles and the next
    // inference reads freed memory (use-after-free crash). A session-exclusive
    // processor's lifetime matches its session, so the reference stays valid.
    config.m_session_exclusive_processor = true;

    return config;
}

}  // namespace

Session::Session(const std::string& json_config_path)
    : m_config_loader(json_config_path)
    , m_anira_context(load_context_config(m_config_loader))
    , m_inference_config(load_inference_config(m_config_loader, json_config_path))
    , m_state_pairs(parse_state_pairs(json_config_path))
    , m_pp_processor(m_inference_config, m_state_pairs)
    , m_inference_handler(m_pp_processor, m_inference_config)
    , m_layout(TensorLayout::from(m_inference_config, m_state_pairs))
    , m_resampler_config(parse_resampler_config(json_config_path)) {}

namespace {

/// For the first upsample tensor (input_size < output_size), compute the
/// extra "decoder" latency anira needs to allocate. Returns std::nullopt
/// when no tensor needs it.
struct DecoderLatency {
    size_t m_tensor_index;
    unsigned int m_samples;
};

std::optional<DecoderLatency> compute_decoder_latency(const TensorLayout& layout,
                                                      size_t buffer_size) {
    for (size_t i = 0; i < layout.m_output_block_sizes.size(); ++i) {
        if (i >= layout.m_input_block_sizes.size()) { break; }
        if (layout.m_input_block_sizes[i] >= layout.m_output_block_sizes[i]) { continue; }
        const size_t out_sz = layout.m_output_block_sizes[i];
        const size_t safety_margin = std::max(buffer_size, out_sz);
        return DecoderLatency{.m_tensor_index = i,
                              .m_samples = static_cast<unsigned int>(out_sz + safety_margin)};
    }
    return std::nullopt;
}

}  // namespace

void Session::prepare(size_t buffer_size, double sample_rate, size_t max_host_block) {
    if (max_host_block == 0) { max_host_block = buffer_size; }
    // Sample-rate conversion: active when the config declares a model rate
    // that differs from the host rate. The whole pipeline below (rate
    // adaptation + anira) then runs in the MODEL-rate domain; the resamplers
    // form a host<->model shim at the Session boundary. Latent/state tensors
    // are per-inference data and never touch the resamplers.
    m_resampling = m_resampler_config.m_model_sample_rate > 0.0 &&
                   std::abs(m_resampler_config.m_model_sample_rate - sample_rate) > 1e-6;
    m_model_per_host = m_resampling ? m_resampler_config.m_model_sample_rate / sample_rate : 1.0;

    // Model-domain block size the pipeline sees per host callback (±1 sample
    // of jitter absorbed by anira's rings; allow_smaller_buffers covers the
    // short blocks).
    const size_t model_buffer =
        m_resampling
            ? static_cast<size_t>(std::ceil(static_cast<double>(buffer_size) * m_model_per_host))
            : buffer_size;
    const double pipeline_rate =
        m_resampling ? m_resampler_config.m_model_sample_rate : sample_rate;

    // Largest per-call sample count process() will see, in the model domain.
    // buffer_size (the inference-pacing block) can be far smaller — 1 for a
    // decoder — so ring margins and scratch must be sized from this instead.
    const size_t model_max_block =
        m_resampling ? static_cast<size_t>(
                           std::ceil(static_cast<double>(max_host_block) * m_model_per_host)) +
                           16
                     : max_host_block;

    anira::HostConfig const host_config{
        static_cast<float>(model_buffer),
        static_cast<float>(pipeline_rate),
        /*allow_smaller_buffers=*/m_resampling,
    };

    if (const auto dl = compute_decoder_latency(m_layout, model_max_block)) {
        m_inference_handler.prepare(host_config, dl->m_samples, dl->m_tensor_index);
    } else {
        m_inference_handler.prepare(host_config);
    }

    // CUSTOM is the only backend guaranteed to exist regardless of compile-time
    // backend selection; a config without model data runs anira's roundtrip
    // processor anyway.
    m_selected_backend = m_inference_config.m_model_data.empty()
                             ? anira::InferenceBackend::CUSTOM
                             : m_inference_config.m_model_data.front().m_backend;
    m_inference_handler.set_inference_backend(m_selected_backend);

    m_rate_adaptor.prepare(m_layout, model_max_block);

    if (m_resampling) {
        const size_t scratch_cap = model_max_block;
        const size_t n_in = m_layout.m_input_block_sizes.size();
        const size_t n_out = m_layout.m_output_block_sizes.size();

        m_in_resamplers.clear();
        m_in_resamplers.resize(n_in);
        m_in_scratch.assign(n_in, {});
        m_in_scratch_ptrs.assign(n_in, {});
        m_in_views.assign(n_in, nullptr);
        m_model_in_counts.assign(n_in, 0);
        for (size_t t = 0; t < n_in; ++t) {
            if (m_layout.m_input_block_sizes[t] == 0) {
                continue;  // state/message tensor
            }
            const size_t channels = m_layout.m_sig_input_channels[t];
            m_in_resamplers[t].prepare(sample_rate,
                                       m_resampler_config.m_model_sample_rate,
                                       channels,
                                       m_resampler_config.quality_for_input(t),
                                       max_host_block);
            m_in_scratch[t].assign(channels, std::vector<float>(scratch_cap, 0.0f));
            m_in_scratch_ptrs[t].resize(channels);
            for (size_t c = 0; c < channels; ++c) {
                m_in_scratch_ptrs[t][c] = m_in_scratch[t][c].data();
            }
        }

        m_out_resamplers.clear();
        m_out_resamplers.resize(n_out);
        m_out_scratch.assign(n_out, {});
        m_out_scratch_ptrs.assign(n_out, {});
        m_out_views.assign(n_out, nullptr);
        m_model_out_counts.assign(n_out, 0);
        m_out_accum.assign(n_out, 0.0);
        for (size_t t = 0; t < n_out; ++t) {
            if (m_layout.m_output_block_sizes[t] == 0) { continue; }
            const size_t channels = m_layout.m_sig_output_channels[t];
            m_out_resamplers[t].prepare(m_resampler_config.m_model_sample_rate,
                                        sample_rate,
                                        channels,
                                        m_resampler_config.quality_for_output(t),
                                        scratch_cap,
                                        /*exact_output_block=*/max_host_block);
            m_out_scratch[t].assign(channels, std::vector<float>(scratch_cap, 0.0f));
            m_out_scratch_ptrs[t].resize(channels);
            for (size_t c = 0; c < channels; ++c) {
                m_out_scratch_ptrs[t][c] = m_out_scratch[t][c].data();
            }
        }
    }
}

size_t Session::get_latency_samples() {
    const size_t pipeline_latency = m_inference_handler.get_latency();
    if (!m_resampling) { return pipeline_latency; }

    // The input resampler is free-running and timeline-aligned: zero stream
    // latency. The pipeline latency converts model → host; the output
    // resampler's exact-mode priming deficit is already in host samples.
    size_t out_latency = 0;
    for (const auto& r : m_out_resamplers) {
        if (r.is_prepared()) {
            out_latency = r.output_latency();
            break;
        }
    }

    const double host_per_model = 1.0 / m_model_per_host;
    return static_cast<size_t>(std::ceil(static_cast<double>(pipeline_latency) * host_per_model)) +
           out_latency;
}

void Session::run_pipeline(const float* const* const* input_data,
                           size_t* num_input_samples,
                           float* const* const* output_data,
                           size_t* num_output_samples) {
    const auto v = m_rate_adaptor.pre_dispatch(m_layout,
                                               input_data,
                                               num_input_samples,
                                               output_data,
                                               num_output_samples);

    // State-passing requires pop-before-push so post_process(N-1) writes
    // fresh state before pre_process(N) reads it.
    if (!m_state_pairs.empty()) {
        m_inference_handler.pop_data(v.m_out_tensors, v.m_out_sample_counts);
        m_inference_handler.push_data(v.m_in_tensors, v.m_in_sample_counts);
    } else {
        m_inference_handler.push_data(v.m_in_tensors, v.m_in_sample_counts);
        m_inference_handler.pop_data(v.m_out_tensors, v.m_out_sample_counts);
    }

    m_rate_adaptor.post_dispatch(m_layout, output_data, num_output_samples);
}

void Session::process(const float* const* const* input_data,
                      size_t* num_input_samples,
                      float* const* const* output_data,
                      size_t* num_output_samples) {
    if (!m_resampling) {
        run_pipeline(input_data, num_input_samples, output_data, num_output_samples);
        return;
    }

    // Host -> model: convert every streamable input tensor into its scratch;
    // state/message tensors pass through untouched (their counts are 0).
    const size_t n_in = m_layout.m_input_block_sizes.size();
    for (size_t t = 0; t < n_in; ++t) {
        if (m_layout.m_input_block_sizes[t] == 0 || num_input_samples[t] == 0) {
            m_in_views[t] = input_data[t];
            m_model_in_counts[t] = num_input_samples[t];
            continue;
        }
        const size_t produced = m_in_resamplers[t].process(input_data[t],
                                                           num_input_samples[t],
                                                           m_in_scratch_ptrs[t].data(),
                                                           m_in_scratch[t][0].size());
        m_in_views[t] = m_in_scratch_ptrs[t].data();
        m_model_in_counts[t] = produced;
    }

    // Model-domain pop counts: Bresenham pacing keeps the long-run pull rate
    // exactly host*model_per_host samples without drift.
    const size_t n_out = m_layout.m_output_block_sizes.size();
    for (size_t t = 0; t < n_out; ++t) {
        if (m_layout.m_output_block_sizes[t] == 0 || num_output_samples[t] == 0) {
            m_out_views[t] = output_data[t];
            m_model_out_counts[t] = num_output_samples[t];
            continue;
        }
        m_out_accum[t] += static_cast<double>(num_output_samples[t]) * m_model_per_host;
        const auto want = static_cast<size_t>(m_out_accum[t]);
        m_out_accum[t] -= static_cast<double>(want);
        m_out_views[t] = m_out_scratch_ptrs[t].data();
        m_model_out_counts[t] = want;
    }

    run_pipeline(m_in_views.data(),
                 m_model_in_counts.data(),
                 m_out_views.data(),
                 m_model_out_counts.data());

    // Model -> host: every streamable output tensor must fill the host buffer
    // exactly.
    for (size_t t = 0; t < n_out; ++t) {
        if (m_layout.m_output_block_sizes[t] == 0 || num_output_samples[t] == 0) { continue; }
        m_out_resamplers[t].process_exact(m_out_scratch_ptrs[t].data(),
                                          m_model_out_counts[t],
                                          output_data[t],
                                          num_output_samples[t]);
    }
}

void Session::set_input(float input, size_t tensor_index, size_t channel) {
    m_pp_processor.set_input(input, tensor_index, channel);
}

float Session::get_output(size_t tensor_index, size_t channel) {
    return m_pp_processor.get_output(tensor_index, channel);
}

}  // namespace anira_tilde
