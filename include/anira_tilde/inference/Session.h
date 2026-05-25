#pragma once

#include <anira/anira.h>
#include <string>

#include "anira_tilde/Exports.h"
#include "anira_tilde/inference/TensorLayout.h"
#include "anira_tilde/rate_adaptation/RateAdaptor.h"
#include "anira_tilde/state_passing/StatePairParser.h"
#include "anira_tilde/state_passing/StatePassingPrePostProcessor.h"

namespace anira_tilde {

class ANIRA_TILDE_API Session {
public:
    explicit Session(std::string json_config_path);
    ~Session() = default;

    void  prepare(size_t buffer_size, double sample_rate);
    void  process(const float* const* const* input_data,  size_t* num_input_samples,
                  float* const* const*       output_data, size_t* num_output_samples);
    void  set_input (float input, size_t tensor_index, size_t channel);
    float get_output(size_t tensor_index, size_t channel);

    size_t get_latency_samples();

    const TensorLayout& layout() const { return m_layout; }

private:
    // Declaration order matches construction order.
    anira::JsonConfigLoader      m_config_loader;
    anira::ContextConfig         m_anira_context;
    anira::InferenceConfig       m_inference_config;
    std::vector<StatePair>       m_state_pairs;        // before m_pp_processor
    StatePassingPrePostProcessor m_pp_processor;       // before m_inference_handler
    anira::InferenceHandler      m_inference_handler;

    anira::InferenceBackend      m_selected_backend;

    TensorLayout                 m_layout;
    RateAdaptor                  m_rate_adaptor;
};

} // namespace anira_tilde
