#pragma once

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
    std::vector<size_t> sig_input_channels;
    std::vector<size_t> sig_output_channels;

    // Per-message-tensor sub-channel counts (one inlet/outlet per entry).
    std::vector<std::vector<size_t>> msg_input_channels;
    std::vector<std::vector<size_t>> msg_output_channels;

    // Per-tensor block size (signal tensors > 0; message/state tensors == 0).
    std::vector<size_t> input_block_sizes;
    std::vector<size_t> output_block_sizes;

    // State-passing wiring.
    std::vector<StatePair> state_pairs;

    // Flat signal-channel → (tensor index, channel-in-tensor). Populated by
    // build_channel_maps(); always sized to total_signal_inputs() /
    // total_signal_outputs() respectively.
    std::vector<size_t> input_channel_to_tensor;
    std::vector<size_t> input_channel_in_tensor;
    std::vector<size_t> output_channel_to_tensor;
    std::vector<size_t> output_channel_in_tensor;

    // Recompute the flat-channel-to-tensor maps from sig_*_channels.
    // Call after sig_input_channels / sig_output_channels are filled.
    void build_channel_maps() {
        input_channel_to_tensor.clear();
        input_channel_in_tensor.clear();
        for (size_t t = 0; t < sig_input_channels.size(); ++t) {
            for (size_t c = 0; c < sig_input_channels[t]; ++c) {
                input_channel_to_tensor.push_back(t);
                input_channel_in_tensor.push_back(c);
            }
        }
        output_channel_to_tensor.clear();
        output_channel_in_tensor.clear();
        for (size_t t = 0; t < sig_output_channels.size(); ++t) {
            for (size_t c = 0; c < sig_output_channels[t]; ++c) {
                output_channel_to_tensor.push_back(t);
                output_channel_in_tensor.push_back(c);
            }
        }
    }

    size_t total_signal_inputs()  const { return input_channel_to_tensor.size(); }
    size_t total_signal_outputs() const { return output_channel_to_tensor.size(); }

    bool is_state_input(size_t tensor_index) const {
        for (const auto& p : state_pairs)
            if (p.input_tensor == tensor_index) return true;
        return false;
    }
    bool is_state_output(size_t tensor_index) const {
        for (const auto& p : state_pairs)
            if (p.output_tensor == tensor_index) return true;
        return false;
    }

    // Dry/wet mixing is only meaningful when every signal in/out tensor
    // pair has matching rate AND matching channel count.
    bool mixing_makes_sense() const {
        for (size_t i = 0;
             i < sig_input_channels.size() && i < sig_output_channels.size(); ++i) {
            if (input_block_sizes[i] != output_block_sizes[i]
                || sig_input_channels[i] != sig_output_channels[i])
                return false;
        }
        return true;
    }
};

} // namespace anira_tilde
