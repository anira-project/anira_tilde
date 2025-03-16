#include "Mixer.h"

Mixer::Mixer()
    : m_sample_rate(0.0), m_buffer_size(0), m_num_channels(0), m_latency_samples(0), m_mix(1.0f) {}

void Mixer::prepare(double sample_rate, size_t buffer_size, size_t num_channels, size_t latency_samples) {
    m_sample_rate = sample_rate;
    m_buffer_size = buffer_size;
    m_num_channels = num_channels;
    m_latency_samples = latency_samples;

    const size_t delay_size = m_latency_samples + m_buffer_size;
    const size_t write_index = 0;
    const size_t read_index = (write_index + delay_size - m_latency_samples) % delay_size;

    m_write_index.assign(m_num_channels, write_index);
    m_read_index.resize(m_num_channels, read_index);
    m_delay_buffer.assign(m_num_channels, std::vector<float>(delay_size, 0.0f));
}

void Mixer::push_dry_sample(float dry_sample, int channel) {
    m_delay_buffer[channel][m_write_index[channel]] = dry_sample;
    m_write_index[channel] = (m_write_index[channel] + 1) % m_delay_buffer[channel].size();
}

float Mixer::mix_wet_sample(float wet_sample, int channel) {
    float delayed_dry_sample = m_delay_buffer[channel][m_read_index[channel]];
    m_read_index[channel] = (m_read_index[channel] + 1) % m_delay_buffer[channel].size();

    float current_mix = m_mix.load(std::memory_order_relaxed);
    return (1.0f - current_mix) * delayed_dry_sample + current_mix * wet_sample;
}

void Mixer::set_mix(float new_mix) {
    m_mix.store(std::clamp(new_mix, 0.0f, 1.0f), std::memory_order_relaxed);
}