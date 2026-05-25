#include "anira_tilde/mixing/Mixer.h"

#include <algorithm>

namespace anira_tilde {

Mixer::Mixer()
    : m_sample_rate(0.0), m_buffer_size(0), m_num_channels(0), m_latency_samples(0), m_mix(1.0f) {}

void Mixer::prepare(double sample_rate, size_t buffer_size, size_t num_channels, size_t latency_samples) {
    m_sample_rate     = sample_rate;
    m_buffer_size     = buffer_size;
    m_num_channels    = num_channels;
    m_latency_samples = latency_samples;

    const size_t delay_size = m_latency_samples + m_buffer_size;
    const size_t write_index = 0;
    const size_t read_index  = (write_index + delay_size - m_latency_samples) % delay_size;

    m_write_index.assign(m_num_channels, write_index);
    m_read_index.assign (m_num_channels, read_index);
    m_last_wet.assign(m_num_channels, 0.0f);
    m_delay_buffer.resize(m_num_channels, delay_size);
    m_delay_buffer.clear();
}

void Mixer::process_channel_block(const float* dry,
                                  const float* wet, size_t wet_period,
                                  float*       out,
                                  size_t frames, size_t channel) {
    for (size_t s = 0; s < frames; ++s) {
        push_dry_sample(dry[s], static_cast<int>(channel));
        const float wet_sample = (wet_period > 0)
            ? wet[s % wet_period]
            : m_last_wet[channel];
        out[s] = mix_wet_sample(wet_sample, static_cast<int>(channel));
    }
    if (wet_period > 0)
        m_last_wet[channel] = wet[wet_period - 1];
}

void Mixer::push_dry_sample(float dry_sample, int channel) {
    const size_t delay_size = m_delay_buffer.get_num_samples();
    m_delay_buffer.get_write_pointer(channel)[m_write_index[channel]] = dry_sample;
    m_write_index[channel] = (m_write_index[channel] + 1) % delay_size;
}

float Mixer::mix_wet_sample(float wet_sample, int channel) {
    const size_t delay_size = m_delay_buffer.get_num_samples();
    const float delayed_dry = m_delay_buffer.get_read_pointer(channel)[m_read_index[channel]];
    m_read_index[channel] = (m_read_index[channel] + 1) % delay_size;

    const float current_mix = m_mix.load(std::memory_order_relaxed);
    return (1.0f - current_mix) * delayed_dry + current_mix * wet_sample;
}

void Mixer::set_mix(float new_mix) {
    m_mix.store(std::clamp(new_mix, 0.0f, 1.0f), std::memory_order_relaxed);
}

} // namespace anira_tilde
