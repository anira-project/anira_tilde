#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "anira_tilde/Exports.h"

namespace anira_tilde {

enum class ResamplerQuality { SincBest, SincMedium, SincFastest, Linear, Hold };

/**
 * @brief Real-time sample-rate converter for one planar multi-channel stream.
 *
 * Wraps one libsamplerate SRC_STATE per channel (planar data, mono states).
 * After prepare() the process path performs no allocation, no locking and no
 * system calls — src_process() is real-time safe on a prepared state.
 *
 * Unconsumed input frames are retained internally between calls, so callers
 * can feed whatever block the host hands them and take whatever the ratio
 * yields (process), or demand an exact output count (process_exact — used on
 * the model→host side where the host expects full buffers; any shortfall
 * during filter priming is zero-filled and accounted for in the reported
 * latency).
 */
class ANIRA_TILDE_API Resampler {
public:
    Resampler();
    ~Resampler();
    Resampler(Resampler&&) noexcept;
    Resampler& operator=(Resampler&&) noexcept;
    Resampler(const Resampler&) = delete;
    Resampler& operator=(const Resampler&) = delete;

    /// Allocates states and scratch for `channels` planar channels converting
    /// src_rate → dst_rate, sized for input blocks up to max_input_block
    /// frames.
    ///
    /// Latency semantics (measured on a scratch instance at prepare time):
    /// libsamplerate keeps the output timeline aligned with the input, so a
    /// free-running process() adds no stream latency — output_latency() is 0.
    /// Exact-output streaming (process_exact with paced input) is different:
    /// the converter withholds part of a filter length of input before it can
    /// fill a whole block, and that one-time shortfall is masked with leading
    /// zeros — a real, constant delay. Pass the exact output block size in
    /// exact_output_block (> 0) to measure it; it is then reported by
    /// output_latency() in output-domain frames.
    void prepare(double src_rate,
                 double dst_rate,
                 size_t channels,
                 ResamplerQuality quality,
                 size_t max_input_block,
                 size_t exact_output_block = 0);

    /// Feed n_in frames, write up to max_out frames; returns frames produced
    /// (identical across channels). Unconsumed input is retained.
    size_t process(const float* const* in, size_t n_in, float* const* out, size_t max_out);

    /// Feed n_in frames and write exactly n_out frames. A shortfall (filter
    /// priming, at stream start only when the caller paces input correctly)
    /// is zero-filled at the head of the output.
    void process_exact(const float* const* in, size_t n_in, float* const* out, size_t n_out);

    /// Measured converter latency in output-domain frames.
    size_t output_latency() const noexcept { return m_latency; }

    double ratio() const noexcept { return m_ratio; }
    bool is_prepared() const noexcept { return !m_channels.empty(); }

    /// Reset converter state (keeps allocations).
    void reset();

private:
    struct ChannelState;  // owns the SRC_STATE; defined in the .cpp

    size_t run_channel(size_t channel_index,
                       const float* in,
                       size_t n_in,
                       float* out,
                       size_t max_out);

    std::vector<std::unique_ptr<ChannelState>> m_channels;
    double m_ratio = 1.0;
    int m_converter = 0;
    size_t m_latency = 0;
};

}  // namespace anira_tilde
