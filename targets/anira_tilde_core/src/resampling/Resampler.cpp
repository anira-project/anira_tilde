#include "anira_tilde/resampling/Resampler.h"

#include <samplerate.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace anira_tilde {

namespace {

int to_src_converter(ResamplerQuality quality) {
    switch (quality) {
        case ResamplerQuality::SincBest: return SRC_SINC_BEST_QUALITY;
        case ResamplerQuality::SincMedium: return SRC_SINC_MEDIUM_QUALITY;
        case ResamplerQuality::SincFastest: return SRC_SINC_FASTEST;
        case ResamplerQuality::Linear: return SRC_LINEAR;
        case ResamplerQuality::Hold: return SRC_ZERO_ORDER_HOLD;
    }
    return SRC_SINC_FASTEST;
}

void check_src(int error, const char* what) {
    if (error != 0) {
        throw std::runtime_error(std::string("[anira_tilde][Resampler] ") + what + ": " +
                                 src_strerror(error));
    }
}

/// Exact-output streaming latency: simulate the paced regime process_exact
/// runs in (Bresenham-paced input, whole output blocks demanded) on a scratch
/// converter and accumulate the priming shortfall until the converter keeps
/// up. libsamplerate's free-running output is timeline-aligned (an impulse at
/// input[0] peaks at output[0] — verified for every converter type), so this
/// shortfall is the converter's only real stream latency, and only the
/// exact-output side has it.
size_t measure_exact_streaming_latency(int converter, double ratio, size_t block) {
    int error = 0;
    SRC_STATE* state = src_new(converter, 1, &error);
    check_src(error, "src_new (latency probe)");

    std::vector<float> in(block + 8, 0.0F);
    std::vector<float> out(block, 0.0F);
    double accum = 0.0;
    size_t deficit = 0;
    size_t stable = 0;
    for (int i = 0; i < 64 && stable < 2; ++i) {
        accum += static_cast<double>(block) / ratio;
        const auto n_in = static_cast<size_t>(accum);
        accum -= static_cast<double>(n_in);

        SRC_DATA data{};
        data.data_in = in.data();
        data.input_frames = static_cast<long>(n_in);
        data.data_out = out.data();
        data.output_frames = static_cast<long>(block);
        data.src_ratio = ratio;
        data.end_of_input = 0;
        if (src_process(state, &data) != 0) { break; }
        const auto gen = static_cast<size_t>(data.output_frames_gen);
        // Note: unconsumed probe input is deliberately dropped between
        // iterations; the paced feed keeps the converter input-starved exactly
        // like the real stream, and src_process consumes greedily anyway.
        deficit += block - gen;
        stable = (gen == block) ? stable + 1 : 0;
    }
    src_delete(state);
    return deficit;
}

}  // namespace

struct Resampler::ChannelState {
    SRC_STATE* m_state = nullptr;
    std::vector<float> m_pending;  ///< retained unconsumed input frames
    size_t m_pending_count = 0;
    std::vector<float> m_feed;  ///< contiguous pending + fresh input scratch

    ~ChannelState() {
        if (m_state != nullptr) { src_delete(m_state); }
    }
};

Resampler::Resampler() = default;
Resampler::~Resampler() = default;
Resampler::Resampler(Resampler&&) noexcept = default;
Resampler& Resampler::operator=(Resampler&&) noexcept = default;

void Resampler::prepare(double src_rate,
                        double dst_rate,
                        size_t channels,
                        ResamplerQuality quality,
                        size_t max_input_block,
                        size_t exact_output_block) {
    assert(src_rate > 0.0 && dst_rate > 0.0 && channels > 0);
    m_ratio = dst_rate / src_rate;
    m_converter = to_src_converter(quality);
    m_latency = exact_output_block > 0
                    ? measure_exact_streaming_latency(m_converter, m_ratio, exact_output_block)
                    : 0;

    // Retained input stays small (SRC consumes greedily), but size generously:
    // one full block plus filter headroom.
    const size_t pending_cap = max_input_block + 512;

    m_channels.clear();
    m_channels.reserve(channels);
    for (size_t c = 0; c < channels; ++c) {
        auto ch = std::make_unique<ChannelState>();
        int error = 0;
        ch->m_state = src_new(m_converter, 1, &error);
        check_src(error, "src_new");
        ch->m_pending.assign(pending_cap, 0.0F);
        ch->m_feed.assign(pending_cap + max_input_block, 0.0F);
        m_channels.push_back(std::move(ch));
    }
}

void Resampler::reset() {
    for (auto& ch : m_channels) {
        src_reset(ch->m_state);
        ch->m_pending_count = 0;
    }
}

size_t Resampler::run_channel(size_t channel_index,
                              const float* in,
                              size_t n_in,
                              float* out,
                              size_t max_out) {
    ChannelState& ch = *m_channels[channel_index];

    // Contiguous view: retained frames first, fresh input after.
    const size_t total = ch.m_pending_count + n_in;
    assert(total <= ch.m_feed.size() && "Resampler fed more than max_input_block");
    if (ch.m_pending_count > 0) {
        std::memcpy(ch.m_feed.data(), ch.m_pending.data(), ch.m_pending_count * sizeof(float));
    }
    if (n_in > 0) { std::memcpy(ch.m_feed.data() + ch.m_pending_count, in, n_in * sizeof(float)); }

    SRC_DATA data{};
    data.data_in = ch.m_feed.data();
    data.input_frames = static_cast<long>(total);
    data.data_out = out;
    data.output_frames = static_cast<long>(max_out);
    data.src_ratio = m_ratio;
    data.end_of_input = 0;
    check_src(src_process(ch.m_state, &data), "src_process");

    const auto used = static_cast<size_t>(data.input_frames_used);
    const size_t remaining = total - used;
    assert(remaining <= ch.m_pending.size());
    if (remaining > 0) {
        std::memmove(ch.m_pending.data(), ch.m_feed.data() + used, remaining * sizeof(float));
    }
    ch.m_pending_count = remaining;

    return static_cast<size_t>(data.output_frames_gen);
}

size_t Resampler::process(const float* const* in, size_t n_in, float* const* out, size_t max_out) {
    size_t produced = 0;
    for (size_t c = 0; c < m_channels.size(); ++c) {
        const size_t gen = run_channel(c, in[c], n_in, out[c], max_out);
        // Identical converter state + identical frame counts per channel keep
        // all channels in lockstep.
        produced = (c == 0) ? gen : std::min(produced, gen);
    }
    return produced;
}

void Resampler::process_exact(const float* const* in,
                              size_t n_in,
                              float* const* out,
                              size_t n_out) {
    for (size_t c = 0; c < m_channels.size(); ++c) {
        const size_t gen = run_channel(c, in[c], n_in, out[c], n_out);
        if (gen < n_out) {
            // Filter priming at stream start: shift what was produced to the
            // tail and lead with silence, so the shortfall appears once as
            // constant latency instead of dropped samples.
            const size_t missing = n_out - gen;
            std::memmove(out[c] + missing, out[c], gen * sizeof(float));
            std::memset(out[c], 0, missing * sizeof(float));
        }
    }
}

}  // namespace anira_tilde
