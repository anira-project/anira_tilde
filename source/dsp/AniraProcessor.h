#pragma once

#include <anira/anira.h>
#include <vector>
#include <string>
#include "StatePairParser.h"
#include "StatePassingPrePostProcessor.h"

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

    const std::vector<StatePair>& get_state_pairs() const { return m_state_pairs; }

private:
    bool is_state_input(size_t tensor_index) const;
    bool is_state_output(size_t tensor_index) const;

    // NOTE: declaration order matches construction order in the initializer list.
    anira::JsonConfigLoader m_config_loader;
    anira::ContextConfig m_anira_context;
    anira::InferenceConfig m_inference_config;
    std::vector<StatePair> m_state_pairs;          // must be before m_pp_processor
    StatePassingPrePostProcessor m_pp_processor;   // must be before m_inference_handler
    anira::InferenceHandler m_inference_handler;

    anira::InferenceBackend m_selected_backend;
};
