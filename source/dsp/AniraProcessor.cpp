#include "AniraProcessor.h"

#include <stdexcept>

static anira::ContextConfig load_context_config(anira::JsonConfigLoader& loader) {
    auto ptr = loader.get_context_config();
    if (!ptr) throw std::runtime_error("Failed to load context config (JSON may be malformed)");
    return std::move(*ptr);
}

static anira::InferenceConfig load_inference_config(anira::JsonConfigLoader& loader, const std::string& path) {
    auto ptr = loader.get_inference_config();
    if (!ptr) throw std::runtime_error("Failed to load inference config from: " + path + " (JSON may be malformed or missing 'inference_config' key)");
    return std::move(*ptr);
}

AniraProcessor::AniraProcessor(std::string json_config_path) :
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

    input_sizes = inSz;
    output_sizes = outSz;

    for (int i = 0; i < (int)inShapes.size(); ++i) {
        if (inSz[i] > 0) {
            inSigCh.push_back(inCh[i]);
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
            inMsgCh.push_back(msg_ch);
        }
    }
    for (int i = 0; i < (int)outShapes.size(); ++i) {
        if (outSz[i] > 0) {
            outSigCh.push_back(outCh[i]);
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
            outMsgCh.push_back(msg_ch);
        }
    }
}

void AniraProcessor::prepare(size_t buffer_size, double sample_rate)
{
    anira::HostConfig host_config {
        static_cast<float>(buffer_size),
        static_cast<float>(sample_rate),
    };

    size_t decoder_index = 0;
    unsigned int decoder_latency = 0;
    bool use_custom_latency = false;

    for (size_t i = 0; i < output_sizes.size(); ++i) {
        if (i < input_sizes.size() && input_sizes[i] < output_sizes[i]) {
            decoder_index = i;
            size_t safety_margin = (buffer_size > output_sizes[i]) ? buffer_size : output_sizes[i];
            decoder_latency = static_cast<unsigned int>(output_sizes[i] + safety_margin);
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
}

size_t AniraProcessor::get_latency_samples()
{
    return m_inference_handler.get_latency();
}

size_t* AniraProcessor::process(const float* const* const* input_data, size_t* num_input_samples, float* const* const* output_data, size_t* num_output_samples)
{
    return m_inference_handler.process(input_data, num_input_samples, output_data, num_output_samples);
}

void AniraProcessor::set_input(const float& input, size_t i, size_t j) {
    m_pp_processor.set_input(input, i, j);
}

float AniraProcessor::get_output(size_t i, size_t j) {
    return m_pp_processor.get_output(i, j);
}

bool AniraProcessor::is_state_input(size_t tensor_index) const {
    for (const auto& pair : m_state_pairs) {
        if (pair.input_tensor == tensor_index) return true;
    }
    return false;
}

bool AniraProcessor::is_state_output(size_t tensor_index) const {
    for (const auto& pair : m_state_pairs) {
        if (pair.output_tensor == tensor_index) return true;
    }
    return false;
}
