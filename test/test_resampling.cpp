// Host<->model sample-rate conversion (resampler_config).
//
// Unit level: the Resampler must preserve frequency content across a rate
// change and report a sane measured latency. Integration level: the sine
// oscillator model (native 44.1 kHz) driven from a 48 kHz host must produce a
// 440 Hz sine in the HOST domain — without conversion the pitch would come
// out at 440 * 48/44.1 ≈ 479 Hz, so a frequency measurement separates a
// working resampler from a bypassed one unambiguously.

#include <chrono>
#include <cmath>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "TestBackends.h"
#include "anira_tilde/inference/Session.h"
#include "anira_tilde/resampling/Resampler.h"

using namespace anira_tilde;
using anira_tilde_test::Backend;

namespace {

constexpr double k_host_rate = 48000.0;
constexpr double k_model_rate = 44100.0;
constexpr size_t k_block = 512;

/// Positive-going zero-crossing frequency estimate over `samples` at `rate`.
double measure_frequency(const std::vector<float>& samples, double rate) {
    size_t crossings = 0;
    size_t first = 0;
    size_t last = 0;
    for (size_t i = 1; i < samples.size(); ++i) {
        if (samples[i - 1] <= 0.0F && samples[i] > 0.0F) {
            if (crossings == 0) first = i;
            last = i;
            ++crossings;
        }
    }
    if (crossings < 2) return 0.0;
    const double periods = static_cast<double>(crossings - 1);
    const double span = static_cast<double>(last - first);
    return periods * rate / span;
}

}  // namespace

// ---------------------------------------------------------------------------
// Resampler unit tests
// ---------------------------------------------------------------------------

TEST(Resampler, PreservesFrequencyUpAndDown) {
    for (const auto& [src, dst] : {std::pair{44100.0, 48000.0}, std::pair{48000.0, 44100.0}}) {
        Resampler resampler;
        resampler.prepare(src, dst, 1, ResamplerQuality::SincFastest, k_block);

        std::vector<float> in(k_block);
        std::vector<float> out_block(k_block * 2);
        std::vector<float> collected;
        double phase = 0.0;
        const double inc = 2.0 * M_PI * 440.0 / src;
        for (int b = 0; b < 60; ++b) {
            for (size_t i = 0; i < k_block; ++i) {
                in[i] = static_cast<float>(std::sin(phase));
                phase += inc;
            }
            const float* in_ptr[1] = {in.data()};
            float* out_ptr[1] = {out_block.data()};
            const size_t produced =
                resampler.process(in_ptr, k_block, out_ptr, out_block.size());
            collected.insert(collected.end(), out_block.begin(),
                             out_block.begin() + static_cast<long>(produced));
        }

        // Skip the converter's own latency, then measure.
        const size_t skip = resampler.output_latency() + k_block;
        ASSERT_GT(collected.size(), skip + 4096);
        const std::vector<float> steady(collected.begin() + static_cast<long>(skip),
                                        collected.end());
        EXPECT_NEAR(measure_frequency(steady, dst), 440.0, 2.0)
            << "ratio " << src << " -> " << dst;
    }
}

TEST(Resampler, LatencyMeasurementSane) {
    // Free-running conversion is timeline-aligned (libsamplerate compensates
    // its filter delay): zero stream latency.
    Resampler free_run;
    free_run.prepare(44100.0, 48000.0, 1, ResamplerQuality::SincFastest, k_block);
    EXPECT_EQ(free_run.output_latency(), 0U);

    // Exact-output streaming primes the filter before it can fill whole
    // blocks: a small, constant, measured deficit.
    Resampler exact;
    exact.prepare(44100.0, 48000.0, 1, ResamplerQuality::SincFastest, k_block, k_block);
    EXPECT_GT(exact.output_latency(), 0U);
    EXPECT_LT(exact.output_latency(), 128U);

    // Zero-order hold has no filter to prime.
    Resampler hold;
    hold.prepare(44100.0, 48000.0, 1, ResamplerQuality::Hold, k_block, k_block);
    EXPECT_LE(hold.output_latency(), 1U);
}

TEST(Resampler, ProcessExactAlwaysFillsTheBuffer) {
    Resampler resampler;
    resampler.prepare(44100.0, 48000.0, 1, ResamplerQuality::SincFastest, k_block);

    // Bresenham-paced input: long-run input rate must be output/ratio.
    double accum = 0.0;
    std::vector<float> in(k_block, 0.25F);
    std::vector<float> out(k_block);
    for (int b = 0; b < 50; ++b) {
        accum += k_block * 44100.0 / 48000.0;
        const size_t n_in = static_cast<size_t>(accum);
        accum -= static_cast<double>(n_in);
        const float* in_ptr[1] = {in.data()};
        float* out_ptr[1] = {out.data()};
        resampler.process_exact(in_ptr, n_in, out_ptr, k_block);
    }
    // Steady state: constant input must yield the constant output.
    EXPECT_NEAR(out[k_block - 1], 0.25F, 1e-3F);
}

// ---------------------------------------------------------------------------
// End-to-end: sine oscillator at a 48 kHz host, model native 44.1 kHz
// ---------------------------------------------------------------------------

class Resampling : public testing::TestWithParam<Backend> {};

TEST_P(Resampling, SineOscillatorPitchCorrectAtForeignHostRate) {
    anira_tilde::Session session(
        anira_tilde_test::json_path("sine_oscillator_resampled_test", GetParam()));
    session.prepare(k_block, k_host_rate);
    ASSERT_TRUE(session.is_resampling());

    const float freq = 440.0F;
    std::vector<float> freq_buf(k_block, freq);
    std::vector<float> audio_buf(k_block);
    const float* freq_ch[1] = {freq_buf.data()};
    const float* dummy_in[1] = {nullptr};
    const float* const* in_ptrs[2] = {freq_ch, dummy_in};
    size_t in_sizes[2] = {k_block, 0};
    float* audio_ch[1] = {audio_buf.data()};
    float* dummy_out[1] = {nullptr};
    float* const* out_ptrs[2] = {audio_ch, dummy_out};
    size_t out_sizes[2] = {k_block, 0};

    auto run_block = [&] {
        std::fill(audio_buf.begin(), audio_buf.end(), 0.0F);
        session.process(in_ptrs, in_sizes, out_ptrs, out_sizes);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    };

    const int warmup =
        static_cast<int>(session.get_latency_samples() / k_block) + 4;
    for (int i = 0; i < warmup; ++i) run_block();

    std::vector<float> collected;
    for (int i = 0; i < 24; ++i) {
        run_block();
        collected.insert(collected.end(), audio_buf.begin(), audio_buf.end());
    }

    // 440 Hz in the HOST domain proves the conversion: a bypassed resampler
    // would land at ~479 Hz.
    EXPECT_NEAR(measure_frequency(collected, k_host_rate), 440.0, 2.0);

    // No discontinuities: phase state must survive the round trip.
    size_t discontinuities = 0;
    const float max_step =
        2.0F * static_cast<float>(M_PI) * freq / static_cast<float>(k_host_rate) * 1.5F;
    for (size_t i = 1; i < collected.size(); ++i) {
        if (std::fabs(collected[i] - collected[i - 1]) > max_step) ++discontinuities;
    }
    EXPECT_EQ(discontinuities, 0U);
}

// The reported latency must be sample-accurate: an impulse through the
// passthrough model (state_accumulator: audio_out == audio_in) must re-emerge
// exactly get_latency_samples() after it went in — through both resamplers
// and the full anira round trip. The sinc filters smear the impulse, so the
// PEAK position is the measured delay; ±2 samples covers the two converters'
// fractional-phase rounding.
TEST_P(Resampling, ReportedLatencyIsSampleAccurate) {
    anira_tilde::Session session(
        anira_tilde_test::json_path("state_accumulator_resampled_test", GetParam()));
    session.prepare(k_block, k_host_rate);
    ASSERT_TRUE(session.is_resampling());

    std::vector<float> in_buf(k_block, 0.0F);
    std::vector<float> out_buf(k_block, 0.0F);
    const float* in_ch[1] = {in_buf.data()};
    const float* dummy_in[1] = {nullptr};
    const float* const* in_ptrs[2] = {in_ch, dummy_in};
    size_t in_sizes[2] = {k_block, 0};
    float* out_ch[1] = {out_buf.data()};
    float* dummy_out[1] = {nullptr};
    float* const* out_ptrs[2] = {out_ch, dummy_out};
    size_t out_sizes[2] = {k_block, 0};

    const size_t reported = session.get_latency_samples();
    const int total_blocks =
        static_cast<int>(reported / k_block) + 8;  // impulse + full drain

    std::vector<float> collected;
    size_t impulse_position = 0;
    for (int b = 0; b < total_blocks; ++b) {
        std::fill(in_buf.begin(), in_buf.end(), 0.0F);
        if (b == 2) {  // a couple of silent blocks first, then the impulse
            in_buf[0] = 1.0F;
            impulse_position = static_cast<size_t>(b) * k_block;
        }
        std::fill(out_buf.begin(), out_buf.end(), 0.0F);
        session.process(in_ptrs, in_sizes, out_ptrs, out_sizes);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        collected.insert(collected.end(), out_buf.begin(), out_buf.end());
    }

    float peak = 0.0F;
    size_t peak_position = 0;
    for (size_t i = 0; i < collected.size(); ++i) {
        if (std::fabs(collected[i]) > peak) {
            peak = std::fabs(collected[i]);
            peak_position = i;
        }
    }
    ASSERT_GT(peak, 0.1F) << "impulse never emerged";

    const auto measured =
        static_cast<long>(peak_position) - static_cast<long>(impulse_position);
    EXPECT_NEAR(static_cast<double>(measured), static_cast<double>(reported), 2.0)
        << "peak at " << peak_position << ", impulse at " << impulse_position;
}

TEST_P(Resampling, InactiveWhenHostMatchesModelRate) {
    anira_tilde::Session session(
        anira_tilde_test::json_path("sine_oscillator_resampled_test", GetParam()));
    session.prepare(k_block, k_model_rate);
    EXPECT_FALSE(session.is_resampling());
}

INSTANTIATE_TEST_SUITE_P(Backends, Resampling,
                         testing::ValuesIn(anira_tilde_test::backends()),
                         anira_tilde_test::param_name);
