#include "anira_tilde/Engine.h"
#include "anira_tilde/rate_adaptation/EffectiveBufferSize.h"

#include <algorithm>

namespace anira_tilde {

bool Engine::load_config(const std::string& json_config_path, std::string* error_out) {
    // Park process() while we swap. Caller must re-invoke prepare() before
    // resuming audio.
    m_ready.store(false, std::memory_order_release);

    if (json_config_path.empty()) {
        if (error_out) *error_out = "empty config path";
        return false;
    }
    try {
        m_session = std::make_unique<Session>(json_config_path);
    } catch (const std::exception& e) {
        m_session.reset();
        if (error_out) *error_out = e.what();
        return false;
    }
    return true;
}

const TensorLayout& Engine::layout() const {
    static const TensorLayout empty;
    return m_session ? m_session->layout() : empty;
}

size_t Engine::latency_samples() const {
    return m_session ? m_session->get_latency_samples() : 0;
}

void Engine::prepare(size_t host_buffer_size, double host_sample_rate) {
    m_host_buffer_size = host_buffer_size;
    float latency = 0.f;

    if (m_session) {
        const TensorLayout& layout = m_session->layout();
        const size_t effective_buffer_size = compute_effective_buffer_size(
            layout.input_block_sizes, layout.output_block_sizes, host_buffer_size);
        m_session->prepare(effective_buffer_size, host_sample_rate, host_buffer_size);
        latency = static_cast<float>(m_session->get_latency_samples());
        prepare_audio_buffers();
    }

    m_mixer.prepare(host_sample_rate, host_buffer_size,
                    m_session ? m_session->layout().total_signal_outputs() : 0,
                    static_cast<size_t>(latency));
    m_ready.store(true, std::memory_order_release);
}

void Engine::prepare_audio_buffers() {
    const TensorLayout& layout = m_session->layout();

    m_input_buffers.clear();
    m_output_buffers.clear();
    m_input_tensor_ptrs.clear();
    m_output_tensor_ptrs.clear();
    m_input_sample_counts.clear();
    m_output_sample_counts.clear();

    m_mixing_disabled = !layout.mixing_makes_sense();
    if (m_mixing_disabled) m_mixer.set_mix(1.0f);

    // One anira::BufferF per input tensor.
    m_input_buffers.reserve(layout.sig_input_channels.size());
    for (size_t t = 0; t < layout.sig_input_channels.size(); ++t) {
        m_input_buffers.emplace_back(layout.sig_input_channels[t], m_host_buffer_size);
        m_input_tensor_ptrs.push_back(m_input_buffers.back().get_array_of_read_pointers());
        m_input_sample_counts.push_back(m_host_buffer_size);
    }

    // One anira::BufferF per output tensor.
    m_output_buffers.reserve(layout.sig_output_channels.size());
    for (size_t t = 0; t < layout.sig_output_channels.size(); ++t) {
        m_output_buffers.emplace_back(layout.sig_output_channels[t], m_host_buffer_size);
        m_output_tensor_ptrs.push_back(m_output_buffers.back().get_array_of_write_pointers());
        m_output_sample_counts.push_back(m_host_buffer_size);
    }

}

void Engine::process_anira() {
    if (!m_session) return;
    std::fill(m_input_sample_counts.begin(),  m_input_sample_counts.end(),  m_host_buffer_size);
    std::fill(m_output_sample_counts.begin(), m_output_sample_counts.end(), m_host_buffer_size);
    m_session->process(m_input_tensor_ptrs.data(),  m_input_sample_counts.data(),
                       m_output_tensor_ptrs.data(), m_output_sample_counts.data());
}

void Engine::process(const float* const* input,  size_t num_input_channels,
                     float* const*       output, size_t num_output_channels,
                     size_t sample_count) {
    const bool ready = m_ready.load(std::memory_order_acquire);
    const TensorLayout& layout = m_session ? m_session->layout() : TensorLayout{};

    if (m_session && ready) {
        size_t flat = 0;
        for (auto& buf : m_input_buffers) {
            for (size_t c = 0; c < buf.get_num_channels(); ++c) {
                if (flat < num_input_channels)
                    std::copy_n(input[flat], sample_count, buf.get_write_pointer(c));
                ++flat;
            }
        }
        process_anira();
    }

    write_mixed_output(input, num_input_channels, output, num_output_channels,
                       sample_count, ready, layout);
}

void Engine::write_mixed_output(const float* const* input,  size_t num_input_channels,
                                float* const*       output, size_t num_output_channels,
                                size_t sample_count, bool ready,
                                const TensorLayout& layout) {
    const size_t copy_out = std::min(num_output_channels, layout.total_signal_outputs());
    for (size_t channel = 0; channel < copy_out; ++channel) {
        const size_t tensor_idx        = layout.output_channel_to_tensor[channel];
        const size_t in_tensor_channel = layout.output_channel_in_tensor[channel];
        const size_t samples_retrieved = m_output_sample_counts[tensor_idx];
        const size_t dry_channel       = (channel < num_input_channels) ? channel : 0;
        const float* wet_ptr = m_output_buffers[tensor_idx].get_read_pointer(in_tensor_channel);

        if (ready) {
            m_mixer.process_channel_block(input[dry_channel],
                                          wet_ptr, samples_retrieved,
                                          output[channel],
                                          sample_count, channel);
        } else {
            for (size_t s = 0; s < sample_count; ++s)
                output[channel][s] = (channel < num_input_channels) ? input[channel][s] : 0.0f;
        }
    }
}

void Engine::set_message_input(size_t tensor_index, const std::vector<float>& values) {
    if (!m_session) return;
    for (size_t i = 0; i < values.size(); ++i)
        m_session->set_input(values[i], tensor_index, i);
}

float Engine::get_message_output(size_t tensor_index, size_t channel) {
    return m_session ? m_session->get_output(tensor_index, channel) : 0.0f;
}

void Engine::set_dry_wet_mix(float mix01) {
    m_mixer.set_mix(mix01);
}

} // namespace anira_tilde
