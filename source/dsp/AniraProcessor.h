#pragma once

#include <anira/anira.h>
#include <vector>
#include <string>

class AniraProcessor {
public:
    AniraProcessor(std::string json_config_path);
    ~AniraProcessor() = default;

    void prepare(size_t buffer_size, double sample_rate);
    size_t get_latency_samples();
    size_t* process(const float* const* const* input_data, size_t* num_input_samples, float* const* const* output_data, size_t* num_output_samples);
    void set_input(const float& input, size_t i, size_t j);
    float get_output(size_t i, size_t j);

    std::vector<size_t> inSigCh;
    std::vector<size_t> outSigCh;
    std::vector<std::vector<size_t>> inMsgCh;
    std::vector<std::vector<size_t>> outMsgCh;

    std::vector<size_t> input_sizes;
    std::vector<size_t> output_sizes;

private:
    anira::JsonConfigLoader m_config_loader;
    anira::ContextConfig m_anira_context;
    anira::InferenceConfig m_inference_config;
    anira::PrePostProcessor m_pp_processor;
    anira::InferenceHandler m_inference_handler;

    anira::InferenceBackend m_selected_backend;
};