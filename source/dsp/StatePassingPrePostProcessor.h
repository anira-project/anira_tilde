#pragma once

#include <anira/anira.h>
#include <vector>

struct StatePair {
    size_t output_tensor;  // global output tensor index (must be non-streamable)
    size_t input_tensor;   // global input tensor index (must be non-streamable)
};

// PrePostProcessor subclass for stateful (e.g. RNN) models where one or more
// output tensors represent state to be passed back as input on the next inference.
//
// After each inference, the state output tensor values are copied to the
// corresponding state input atomics so pre_process() picks them up next time.
// This all runs in the inference thread, so no extra synchronisation is needed.
//
// State tensor indices use the same global indexing as InferenceConfig:
//   - output_tensor indexes into tensor_shape[0].output_shape
//   - input_tensor  indexes into tensor_shape[0].input_shape
// Both must refer to non-streamable tensors (postprocess_output_size /
// preprocess_input_size == 0).
//
// State tensors are NOT exposed as Max message inlets/outlets; they are
// managed entirely internally. Initial state is zeros.
class StatePassingPrePostProcessor : public anira::PrePostProcessor {
public:
    StatePassingPrePostProcessor(anira::InferenceConfig& config,
                                  const std::vector<StatePair>& state_pairs)
        : anira::PrePostProcessor(config), m_state_pairs(state_pairs) {
        // Zero-initialize state input tensors so the first inference gets a
        // clean initial state rather than whatever malloc left in memory.
        for (const auto& pair : m_state_pairs) {
            const size_t size = m_inference_config.get_tensor_input_size()[pair.input_tensor];
            for (size_t j = 0; j < size; ++j) {
                set_input(0.0f, pair.input_tensor, j);
            }
        }
    }

    void post_process(std::vector<anira::BufferF>& input,
                      std::vector<anira::RingBuffer>& output,
                      anira::InferenceBackend backend) override {
        // Let the base class store inference outputs to atomic storage.
        anira::PrePostProcessor::post_process(input, output, backend);

        // Feed each state output back to its corresponding state input so that
        // the next call to pre_process() uses the freshly computed state.
        for (const auto& pair : m_state_pairs) {
            const size_t size = m_inference_config.get_tensor_output_size()[pair.output_tensor];
            for (size_t j = 0; j < size; ++j) {
                set_input(get_output(pair.output_tensor, j), pair.input_tensor, j);
            }
        }
    }

private:
    std::vector<StatePair> m_state_pairs;
};
