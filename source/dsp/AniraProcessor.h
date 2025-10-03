#pragma once

#include <anira/anira.h>
#include <c74_min.h>

class AniraProcessor {
public:
    AniraProcessor(std::string json_config_path) :
        m_config_loader(json_config_path),
        m_anira_context(std::move(*m_config_loader.get_context_config())),
        m_inference_config(std::move(*m_config_loader.get_inference_config())),
        m_pp_processor(m_inference_config),
        m_inference_handler(m_pp_processor, m_inference_config)
    {
        auto processing_spec = m_inference_config.m_processing_spec;

        for (int i = 0; i < processing_spec.m_preprocess_input_channels.size(); ++i) {
            c74::max::post("Input Tensor %i Channels: %i", i, processing_spec.m_preprocess_input_channels[i]);
        }

        for (int i = 0; i < processing_spec.m_postprocess_output_channels.size(); ++i) {
            c74::max::post("Output Tensor %i Channels: %i", i, processing_spec.m_postprocess_output_channels[i]);
        }

        for (int i = 0; i < processing_spec.m_preprocess_input_size.size(); ++i) {
            c74::max::post("Input Tensor %i Preprocess Size: %i", i, processing_spec.m_preprocess_input_size[i]);
        }

        for (int i = 0; i < processing_spec.m_postprocess_output_size.size(); ++i) {
            c74::max::post("Output Tensor %i Postprocess Size: %i", i, processing_spec.m_postprocess_output_size[i]);
        }
        
        std::vector<size_t> inShapes;
        std::vector<size_t> outShapes;

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

        for (int i = 0; i < inShapes.size(); ++i) {
            // determine if signal or message
            if (inSz[i] > 0) {
                // audio signal
                inSigCh.push_back(inCh[i]);
            } else {
                // message
                size_t channels = inCh[i];
                size_t total_elems = inShapes[i];
                std::vector<size_t> msg_ch;
                if (channels == 0) channels = 1; // avoid division by zero
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
        for (int i = 0; i < outShapes.size(); ++i) {
            // determine if signal or message
            if (outSz[i] > 0) {
                // audio signal
                outSigCh.push_back(outCh[i]);
            } else {
                // message
                size_t channels = outCh[i];
                size_t total_elems = outShapes[i];
                std::vector<size_t> msg_ch;
                if (channels == 0) channels = 1; // avoid division by zero
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

    ~AniraProcessor() = default;

    void prepare(size_t buffer_size, double sample_rate) 
    {
        anira::HostConfig host_config {
            static_cast<float>(buffer_size),
            static_cast<float>(sample_rate),
        };

        m_inference_handler.prepare(host_config);

        m_selected_backend = anira::InferenceBackend::LIBTORCH;
        m_inference_handler.set_inference_backend(m_selected_backend);
    }
    
    size_t get_latency_samples() 
    {
        return m_inference_handler.get_latency();
    }

    void process(float** inputs, float** outputs, size_t sample_count) 
    {
        m_inference_handler.process(inputs, sample_count);
    }
    
    void set_input(const float& input, size_t i, size_t j) {
        m_pp_processor.set_input(input, i, j);
    }
    
    float get_output(size_t i, size_t j) {
        return m_pp_processor.get_output(i, j);
    }

    std::vector<size_t> inSigCh;
    std::vector<size_t> outSigCh;
    std::vector<std::vector<size_t>> inMsgCh;
    std::vector<std::vector<size_t>> outMsgCh;


private:
    anira::JsonConfigLoader m_config_loader;
    anira::ContextConfig m_anira_context;
    anira::InferenceConfig m_inference_config;
    anira::PrePostProcessor m_pp_processor;
    anira::InferenceHandler m_inference_handler;

    anira::InferenceBackend m_selected_backend;

};