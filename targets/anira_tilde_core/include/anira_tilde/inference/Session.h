#pragma once

#include <anira/anira.h>

#include <string>
#include <vector>

#include "anira_tilde/Exports.h"
#include "anira_tilde/inference/TensorLayout.h"
#include "anira_tilde/rate_adaptation/RateAdaptor.h"
#include "anira_tilde/resampling/Resampler.h"
#include "anira_tilde/resampling/ResamplerConfigParser.h"
#include "anira_tilde/state_passing/StatePairParser.h"
#include "anira_tilde/state_passing/StatePassingPrePostProcessor.h"

namespace anira_tilde {

class ANIRA_TILDE_API Session {
public:
    explicit Session(const std::string& json_config_path);
    ~Session() = default;

    /// buffer_size is the effective inference-pacing block (what the Engine
    /// computes per tensor ratios); max_host_block is the largest per-call
    /// sample count process() may receive on any streamable tensor (the host
    /// buffer size). Direct callers usually pass the same value for both —
    /// the default 0 means "same as buffer_size".
    void prepare(size_t buffer_size, double sample_rate, size_t max_host_block = 0);
    void process(const float* const* const* input_data,
                 size_t* num_input_samples,
                 float* const* const* output_data,
                 size_t* num_output_samples);
    void set_input(float input, size_t tensor_index, size_t channel);
    float get_output(size_t tensor_index, size_t channel);

    size_t get_latency_samples();

    const TensorLayout& layout() const { return m_layout; }

    /// True when resampler_config.m_model_sample_rate is set and differs from
    /// the prepared host rate — every streamable tensor is then converted
    /// host→model on the way in and model→host on the way out.
    bool is_resampling() const noexcept { return m_resampling; }

private:
    /// The model-rate pipeline: rate adaptation + anira push/pop (with the
    /// state-passing pop-before-push ordering). Views are in the model domain
    /// when resampling, in the host domain otherwise.
    void run_pipeline(const float* const* const* input_data,
                      size_t* num_input_samples,
                      float* const* const* output_data,
                      size_t* num_output_samples);

    // Declaration order matches construction order.
    anira::JsonConfigLoader m_config_loader;
    anira::ContextConfig m_anira_context;
    anira::InferenceConfig m_inference_config;
    std::vector<StatePair> m_state_pairs;         // before m_pp_processor
    StatePassingPrePostProcessor m_pp_processor;  // before m_inference_handler
    anira::InferenceHandler m_inference_handler;

    anira::InferenceBackend m_selected_backend;

    TensorLayout m_layout;
    RateAdaptor m_rate_adaptor;

    // ---- host<->model sample-rate conversion (resampler_config) ----
    ResamplerConfig m_resampler_config;
    bool m_resampling = false;
    double m_model_per_host = 1.0;  // model_rate / host_rate

    std::vector<Resampler> m_in_resamplers;   // per input tensor (streamable only prepared)
    std::vector<Resampler> m_out_resamplers;  // per output tensor

    // Model-domain scratch, [tensor][channel][frame], plus the pointer arrays
    // shaped like the host's input_data/output_data. Allocated in prepare().
    std::vector<std::vector<std::vector<float>>> m_in_scratch;
    std::vector<std::vector<std::vector<float>>> m_out_scratch;
    std::vector<std::vector<float*>> m_in_scratch_ptrs;
    std::vector<std::vector<float*>> m_out_scratch_ptrs;
    std::vector<const float* const*> m_in_views;
    std::vector<float* const*> m_out_views;
    std::vector<size_t> m_model_in_counts;
    std::vector<size_t> m_model_out_counts;
    std::vector<double> m_out_accum;  // Bresenham pacing per out tensor
};

}  // namespace anira_tilde
