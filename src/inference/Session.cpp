#include "anira_tilde/inference/Session.h"

#include <filesystem>
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
    m_inference_handler(m_pp_processor, m_inference_config)
{
    std::vector<size_t> inShapes;
    std::vector<size_t> outShapes;
    auto processing_spec = m_inference_config.m_processing_spec;

    for (int i = 0; i < m_inference_config.m_tensor_shape[0].m_tensor_input_shape.size(); ++i) {
        size_t size = 1;
        for (int j = 0; j < m_inference_config.m_tensor_shape[0].m_tensor_input_shape[i].size(); ++j) {
            size *= static_cast<size_t>(m_inference_config.m_tensor_shape[0].m_tensor_input_shape[i][j]);
        }
        inShapes.emplace_back(size);
    }

    for (int i = 0; i < m_inference_config.m_tensor_shape[0].m_tensor_output_shape.size(); ++i) {
        size_t size = 1;
        for (int j = 0; j < m_inference_config.m_tensor_shape[0].m_tensor_output_shape[i].size(); ++j) {
            size *= static_cast<size_t>(m_inference_config.m_tensor_shape[0].m_tensor_output_shape[i][j]);
        }
        outShapes.emplace_back(size);
    }

    auto inSz  = processing_spec.m_preprocess_input_size;
    auto inCh  = processing_spec.m_preprocess_input_channels;
    auto outSz = processing_spec.m_postprocess_output_size;
    auto outCh = processing_spec.m_postprocess_output_channels;

    m_layout.input_block_sizes = inSz;
    m_layout.output_block_sizes = outSz;

    for (int i = 0; i < (int)inShapes.size(); ++i) {
        if (inSz[i] > 0) {
            m_layout.sig_input_channels.push_back(inCh[i]);
        } else if (is_state_input(static_cast<size_t>(i))) {
            // State input tensor: managed internally, not exposed as a Max inlet.
        } else {
            size_t channels = inCh[i];
            size_t total_elems = inShapes[i];
            std::vector<size_t> msg_ch;
            if (channels == 0) channels = 1;
            if (total_elems % channels != 0) {
                throw std::runtime_error("Inconsistent message channel configuration");
            }
            size_t elems_per_ch = total_elems / channels;
            for (size_t c = 0; c < channels; ++c) {
                msg_ch.push_back(elems_per_ch);
            }
            m_layout.msg_input_channels.push_back(msg_ch);
        }
    }
    for (int i = 0; i < (int)outShapes.size(); ++i) {
        if (outSz[i] > 0) {
            m_layout.sig_output_channels.push_back(outCh[i]);
        } else if (is_state_output(static_cast<size_t>(i))) {
            // State output tensor: fed back internally, not exposed as a Max outlet.
        } else {
            size_t channels = outCh[i];
            size_t total_elems = outShapes[i];
            std::vector<size_t> msg_ch;
            if (channels == 0) channels = 1;
            if (total_elems % channels != 0) {
                throw std::runtime_error("Inconsistent message channel configuration");
            }
            size_t elems_per_ch = total_elems / channels;
            for (size_t c = 0; c < channels; ++c) {
                msg_ch.push_back(elems_per_ch);
            }
            m_layout.msg_output_channels.push_back(msg_ch);
        }
    }

    m_layout.state_pairs = m_state_pairs;
    m_layout.build_channel_maps();
}

void Session::prepare(size_t buffer_size, double sample_rate)
{
    anira::HostConfig host_config {
        static_cast<float>(buffer_size),
        static_cast<float>(sample_rate),
    };

    size_t decoder_index = 0;
    unsigned int decoder_latency = 0;
    bool use_custom_latency = false;

    for (size_t i = 0; i < m_layout.output_block_sizes.size(); ++i) {
        if (i < m_layout.input_block_sizes.size() && m_layout.input_block_sizes[i] < m_layout.output_block_sizes[i]) {
            decoder_index = i;
            size_t safety_margin = (buffer_size > m_layout.output_block_sizes[i]) ? buffer_size : m_layout.output_block_sizes[i];
            decoder_latency = static_cast<unsigned int>(m_layout.output_block_sizes[i] + safety_margin);
            use_custom_latency = true;
            break;
        }
    }

    if (use_custom_latency) {
        m_inference_handler.prepare(host_config, decoder_latency, decoder_index);
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
    m_rate_adaptor.pre_dispatch(m_layout,
                                input_data,  num_input_samples,
                                output_data, num_output_samples);

    auto* views_in    = m_rate_adaptor.input_views();
    auto* views_out   = m_rate_adaptor.output_views();
    auto* views_in_n  = m_rate_adaptor.input_view_sample_counts();
    auto* views_out_n = m_rate_adaptor.output_view_sample_counts();

    // State-passing requires pop-before-push so post_process(N-1) writes
    // fresh state before pre_process(N) reads it.
    if (!m_state_pairs.empty()) {
        m_inference_handler.pop_data (views_out, views_out_n);
        m_inference_handler.push_data(views_in,  views_in_n);
    } else {
        m_inference_handler.push_data(views_in,  views_in_n);
        m_inference_handler.pop_data (views_out, views_out_n);
    }

    m_rate_adaptor.post_dispatch(m_layout, output_data, num_output_samples);
}

void Session::set_input(float input, size_t tensor_index, size_t channel) {
    m_pp_processor.set_input(input, tensor_index, channel);
}

float Session::get_output(size_t tensor_index, size_t channel) {
    return m_pp_processor.get_output(tensor_index, channel);
}

bool Session::is_state_input(size_t tensor_index) const {
    for (const auto& pair : m_state_pairs) {
        if (pair.input_tensor == tensor_index) return true;
    }
    return false;
}

bool Session::is_state_output(size_t tensor_index) const {
    for (const auto& pair : m_state_pairs) {
        if (pair.output_tensor == tensor_index) return true;
    }
    return false;
}

} // namespace anira_tilde
