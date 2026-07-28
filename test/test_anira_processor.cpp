#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <thread>
#include <vector>

#include "TestBackends.h"
#include "anira_tilde/inference/Session.h"

using namespace anira_tilde;

#ifndef SINE_OSC_RELATIVE_JSON_PATH
#error "SINE_OSC_RELATIVE_JSON_PATH must be defined via CMake"
#endif

using anira_tilde_test::Backend;

// Runs once per compiled-in backend (see TestBackends.h); LibTorch coverage
// exists only when the build enables ANIRA_WITH_LIBTORCH.
class AniraSession : public testing::TestWithParam<Backend> {};

static constexpr size_t k_ap_sine_signal_size = 512;
static constexpr float k_ap_sample_rate = 44100.0f;

// Simulate Max's audio callback pattern: process() is called on a fixed
// hardware clock without waiting for the previous inference to complete.
// With 50ms between calls a fast model will always finish in time, so any
// continuity failure is due to the processing ORDER within process(), not
// to a genuine timing race.
//
// Root cause of the bug this test catches:
//   InferenceManager::process() called new_data_submitted (pre_process —
//   reads state) BEFORE new_data_request (post_process — writes state).
//   pre_process(N) therefore always read state written by post_process(N-2),
//   not post_process(N-1): state was perpetually one inference stale.
//
// The fix in anira_tilde::Session::process(): call pop_data (→ new_data_request
// → post_process(N-1) writes fresh state) BEFORE push_data (→
// new_data_submitted → pre_process(N) reads it).
TEST_P(AniraSession, MaxLikeCallbackPatternAreContinuous) {
    anira_tilde::Session proc(anira_tilde_test::json_path("sine_oscillator_test", GetParam()));
    proc.prepare(k_ap_sine_signal_size, k_ap_sample_rate);

    const float freq = 440.0f;
    const float max_step = 2.0f * std::numbers::pi_v<float> * freq / k_ap_sample_rate;

    // Tensor 0 (freq, streamable): provide kAPSineSignalSize samples per call.
    // Tensor 1 (phase, state):     managed internally — pass 0 samples.
    std::vector<float> freq_buf(k_ap_sine_signal_size, freq);
    std::array<const float*, 1> freq_ch = {freq_buf.data()};
    std::array<const float*, 1> dummy_in = {nullptr};  // never dereferenced
    std::array<const float* const*, 2> in_ptrs = {freq_ch.data(), dummy_in.data()};
    std::array<size_t, 2> in_sizes = {k_ap_sine_signal_size, 0};

    std::vector<float> audio_buf(k_ap_sine_signal_size);
    std::array<float*, 1> audio_ch = {audio_buf.data()};
    std::array<float*, 1> dummy_out = {nullptr};  // never dereferenced
    std::array<float* const*, 2> out_ptrs = {audio_ch.data(), dummy_out.data()};
    std::array<size_t, 2> out_sizes = {k_ap_sine_signal_size, 0};

    // Each "callback": call process() and capture audio output.
    static constexpr auto k_callback_interval = std::chrono::milliseconds(50);

    auto run_callback = [&](std::vector<float>& out) {
        audio_buf.assign(k_ap_sine_signal_size, 0.0f);
        proc.process(in_ptrs.data(), in_sizes.data(), out_ptrs.data(), out_sizes.data());
        out.assign(audio_buf.begin(), audio_buf.end());
        std::this_thread::sleep_for(k_callback_interval);
    };

    // Warm up to move past latency-padding zeros.
    int const warmup = static_cast<int>(proc.get_latency_samples() / k_ap_sine_signal_size) + 2;
    std::vector<float> discard;
    for (int i = 0; i < warmup; ++i) { run_callback(discard); }

    // Capture consecutive blocks the Max way (no explicit wait between calls).
    static constexpr int k_num_blocks = 4;
    std::vector<std::vector<float>> blocks(k_num_blocks);
    for (int b = 0; b < k_num_blocks; ++b) { run_callback(blocks[b]); }

    // Check sample-to-sample continuity within each block and at every
    // block boundary.  A discontinuity at a boundary means pre_process(N)
    // read a stale phase and the sine wave reset to an earlier position.
    auto check_step = [&](float a, float b, size_t idx) {
        EXPECT_LE(std::abs(b - a), max_step * 1.1f) << "discontinuity at sample " << idx;
    };
    for (int blk = 0; blk < k_num_blocks; ++blk) {
        for (size_t i = 1; i < k_ap_sine_signal_size; ++i) {
            check_step(blocks[blk][i - 1],
                       blocks[blk][i],
                       static_cast<size_t>(blk) * k_ap_sine_signal_size + i);
        }
        if (blk + 1 < k_num_blocks) {
            check_step(blocks[blk].back(),
                       blocks[blk + 1].front(),
                       static_cast<size_t>(blk + 1) * k_ap_sine_signal_size);
        }
    }
}

// Not backend-parameterized: the committed fixture uses the always-available
// ONNX backend; what is under test is the relative-path resolution.
TEST(AniraSessionPaths, RelativeModelPathLoads) {
    // Verifies that model_path values relative to the JSON config file are resolved correctly.
    EXPECT_NO_THROW(anira_tilde::Session const proc(SINE_OSC_RELATIVE_JSON_PATH));
}

INSTANTIATE_TEST_SUITE_P(Backends,
                         AniraSession,
                         testing::ValuesIn(anira_tilde_test::backends()),
                         anira_tilde_test::param_name);
