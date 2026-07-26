#include "anira_tilde/inference/TensorLayout.h"

#include <stdexcept>

namespace anira_tilde {

namespace {

/// Product of one tensor's shape array.
size_t element_count(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (int64_t d : shape) n *= static_cast<size_t>(d);
    return n;
}

/// Split a non-streamable tensor into per-channel sub-counts.
std::vector<size_t> message_subchannels(size_t total_elems, size_t declared_channels) {
    size_t channels = declared_channels == 0 ? 1 : declared_channels;
    if (total_elems % channels != 0)
        throw std::runtime_error("Inconsistent message channel configuration");
    const size_t elems_per_ch = total_elems / channels;
    return std::vector<size_t>(channels, elems_per_ch);
}

bool tensor_referenced(const std::vector<StatePair>& pairs,
                       size_t tensor_index, bool side_is_input) {
    for (const auto& p : pairs)
        if ((side_is_input ? p.input_tensor : p.output_tensor) == tensor_index)
            return true;
    return false;
}

} // namespace

TensorLayout TensorLayout::from(const anira::InferenceConfig& cfg,
                                std::vector<StatePair>        state_pairs) {
    TensorLayout layout;
    layout.state_pairs        = std::move(state_pairs);
    layout.input_block_sizes  = cfg.m_processing_spec.m_preprocess_input_size;
    layout.output_block_sizes = cfg.m_processing_spec.m_postprocess_output_size;

    const auto& in_channels  = cfg.m_processing_spec.m_preprocess_input_channels;
    const auto& out_channels = cfg.m_processing_spec.m_postprocess_output_channels;
    const auto& in_shape     = cfg.m_tensor_shape[0].m_tensor_input_shape;
    const auto& out_shape    = cfg.m_tensor_shape[0].m_tensor_output_shape;

    for (size_t i = 0; i < in_shape.size(); ++i) {
        if (layout.input_block_sizes[i] > 0) {
            layout.sig_input_channels.push_back(in_channels[i]);
        } else if (tensor_referenced(layout.state_pairs, i, /*side_is_input=*/true)) {
            // State input tensor: managed internally, no Max inlet.
        } else {
            layout.msg_input_channels.push_back(
                message_subchannels(element_count(in_shape[i]), in_channels[i]));
        }
    }

    for (size_t i = 0; i < out_shape.size(); ++i) {
        if (layout.output_block_sizes[i] > 0) {
            layout.sig_output_channels.push_back(out_channels[i]);
        } else if (tensor_referenced(layout.state_pairs, i, /*side_is_input=*/false)) {
            // State output tensor: fed back internally, no Max outlet.
        } else {
            layout.msg_output_channels.push_back(
                message_subchannels(element_count(out_shape[i]), out_channels[i]));
        }
    }

    layout.build_channel_maps();
    return layout;
}

void TensorLayout::build_channel_maps() {
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

bool TensorLayout::is_state_tensor(const std::vector<StatePair>& pairs,
                                   size_t tensor_index, bool side_is_input) const {
    return tensor_referenced(pairs, tensor_index, side_is_input);
}

} // namespace anira_tilde
