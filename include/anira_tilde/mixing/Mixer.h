#pragma once

#include <anira/anira.h>
#include <atomic>
#include <cstddef>
#include <vector>

#include "anira_tilde/Exports.h"

namespace anira_tilde {

class ANIRA_TILDE_API Mixer {
public:
    Mixer();

    void prepare(double sample_rate, size_t buffer_size, size_t num_channels, size_t latency_samples);

    /**
     * @brief Mix one block of one channel.
     *
     * Pushes each dry sample into the latency-aligning delay line and writes
     * mix(dry_delayed, wet) into out. Handles rate adaptation: if
     * wet_period > 0 the wet sample is read as wet[s % wet_period]; if
     * wet_period == 0 the mixer falls back to the last valid wet sample
     * cached for this channel.
     */
    void process_channel_block(const float* dry,
                               const float* wet, size_t wet_period,
                               float*       out,
                               size_t frames, size_t channel);

    // Per-sample API (kept for unit tests / fine-grained use).
    void  push_dry_sample(float dry_sample, int channel);
    float mix_wet_sample (float wet_sample, int channel);

    void   set_mix(float new_mix);
    size_t get_latency() const { return m_latency_samples; }

private:
    anira::Buffer<float> m_delay_buffer;          // channels × (latency + buffer_size)
    std::vector<size_t>  m_write_index;
    std::vector<size_t>  m_read_index;
    std::vector<float>   m_last_wet;              // per-channel fallback when wet_period == 0

    double m_sample_rate;
    size_t m_buffer_size;
    size_t m_num_channels;
    size_t m_latency_samples;

    std::atomic<float> m_mix;
};

} // namespace anira_tilde
