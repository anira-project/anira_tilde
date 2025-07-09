#ifndef ANIRA_EXTERNAL_DRY_WET_MIXER_H
#define ANIRA_EXTERNAL_DRY_WET_MIXER_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <atomic>

class Mixer {
public:
    Mixer();

    void prepare(double sample_rate, size_t buffer_size, size_t num_channels, size_t latency_samples);

    void push_dry_sample(float dry_sample, int channel);
    float mix_wet_sample(float wet_sample, int channel);

    void set_mix(float new_mix);
    size_t get_latency() const { return m_latency_samples; }

private:
    std::vector<std::vector<float>> m_delay_buffer;
    std::vector<size_t> m_write_index;
    std::vector<size_t> m_read_index;

    double m_sample_rate;
    size_t m_buffer_size;
    size_t m_num_channels;
    size_t m_latency_samples;

    std::atomic<float> m_mix;
};

#endif //ANIRA_EXTERNAL_DRY_WET_MIXER_H

