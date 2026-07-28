#pragma once

#include <anira/anira.h>

#include <cstddef>
#include <vector>

#include "anira_tilde/state_passing/StatePairParser.h"

namespace anira_tilde {

/**
 * @brief Describes the model's tensor I/O shape and the precomputed
 *        lookups that hosts (and the Engine) need to drive it.
 *
 * Populated once when a Session loads a JSON config. Read-only thereafter.
 */
struct TensorLayout {
    // Per-signal-tensor channel counts.
    std::vector<size_t> m_sig_input_channels;
    std::vector<size_t> m_sig_output_channels;

    // Per-message-tensor sub-channel counts (one inlet/outlet per entry).
    std::vector<std::vector<size_t>> m_msg_input_channels;
    std::vector<std::vector<size_t>> m_msg_output_channels;

    // Per-tensor block size (signal tensors > 0; message/state tensors == 0).
    std::vector<size_t> m_input_block_sizes;
    std::vector<size_t> m_output_block_sizes;

    // State-passing wiring.
    std::vector<StatePair> m_state_pairs;

    // Flat signal-channel → (tensor index, channel-in-tensor). Built by
    // build_channel_maps(); sized to total_signal_inputs()/outputs().
    std::vector<size_t> m_input_channel_to_tensor;
    std::vector<size_t> m_input_channel_in_tensor;
    std::vector<size_t> m_output_channel_to_tensor;
    std::vector<size_t> m_output_channel_in_tensor;

    /// Construct a layout from anira's InferenceConfig and the parsed
    /// state pairs. Builds sig/msg channel vectors, block sizes, state
    /// pairs, and the channel→tensor lookups. Throws on malformed config
    /// (e.g. message tensor whose element count isn't divisible by its
    /// channel count).
    static TensorLayout from(const anira::InferenceConfig& cfg, std::vector<StatePair> state_pairs);

    /// (Re)build the flat-channel-to-tensor lookups from sig_*_channels.
    void build_channel_maps();

    size_t total_signal_inputs() const { return m_input_channel_to_tensor.size(); }
    size_t total_signal_outputs() const { return m_output_channel_to_tensor.size(); }

    bool is_state_tensor(const std::vector<StatePair>& pairs,
                         size_t tensor_index,
                         bool side_is_input) const;

    bool is_state_input(size_t tensor_index) const {
        return is_state_tensor(m_state_pairs, tensor_index, true);
    }
    bool is_state_output(size_t tensor_index) const {
        return is_state_tensor(m_state_pairs, tensor_index, false);
    }
};

}  // namespace anira_tilde
