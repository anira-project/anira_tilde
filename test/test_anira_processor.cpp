#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <cmath>
#include "AniraProcessor.h"

#ifndef SINE_OSC_JSON_PATH
#error "SINE_OSC_JSON_PATH must be defined via CMake"
#endif
#ifndef SINE_OSC_RELATIVE_JSON_PATH
#error "SINE_OSC_RELATIVE_JSON_PATH must be defined via CMake"
#endif

#ifdef USE_LIBTORCH

static constexpr size_t kAPSineSignalSize = 512;
static constexpr float  kAPSampleRate     = 44100.0f;

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
// The fix in AniraProcessor::process(): call pop_data (→ new_data_request
// → post_process(N-1) writes fresh state) BEFORE push_data (→
// new_data_submitted → pre_process(N) reads it).
TEST(AniraProcessor, MaxLikeCallbackPatternAreContinuous) {
    AniraProcessor proc(SINE_OSC_JSON_PATH);
    proc.prepare(kAPSineSignalSize, kAPSampleRate);

    const float freq     = 440.0f;
    const float max_step = 2.0f * static_cast<float>(M_PI) * freq / kAPSampleRate;

    // Tensor 0 (freq, streamable): provide kAPSineSignalSize samples per call.
    // Tensor 1 (phase, state):     managed internally — pass 0 samples.
    std::vector<float> freq_buf(kAPSineSignalSize, freq);
    const float* freq_ch[1]   = { freq_buf.data() };
    const float* dummy_in[1]  = { nullptr };          // never dereferenced
    const float* const* in_ptrs[2] = { freq_ch, dummy_in };
    size_t in_sizes[2] = { kAPSineSignalSize, 0 };

    std::vector<float> audio_buf(kAPSineSignalSize);
    float* audio_ch[1]  = { audio_buf.data() };
    float* dummy_out[1] = { nullptr };                // never dereferenced
    float* const* out_ptrs[2] = { audio_ch, dummy_out };
    size_t out_sizes[2] = { kAPSineSignalSize, 0 };

    // Each "callback": call process() and capture audio output.
    static constexpr auto kCallbackInterval = std::chrono::milliseconds(50);

    auto run_callback = [&](std::vector<float>& out) {
        audio_buf.assign(kAPSineSignalSize, 0.0f);
        proc.process(in_ptrs, in_sizes, out_ptrs, out_sizes);
        out.assign(audio_buf.begin(), audio_buf.end());
        std::this_thread::sleep_for(kCallbackInterval);
    };

    // Warm up to move past latency-padding zeros.
    int warmup = static_cast<int>(proc.get_latency_samples() / kAPSineSignalSize) + 2;
    std::vector<float> discard;
    for (int i = 0; i < warmup; ++i)
        run_callback(discard);

    // Capture consecutive blocks the Max way (no explicit wait between calls).
    static constexpr int kNumBlocks = 4;
    std::vector<std::vector<float>> blocks(kNumBlocks);
    for (int b = 0; b < kNumBlocks; ++b)
        run_callback(blocks[b]);

    // Check sample-to-sample continuity within each block and at every
    // block boundary.  A discontinuity at a boundary means pre_process(N)
    // read a stale phase and the sine wave reset to an earlier position.
    auto check_step = [&](float a, float b, size_t idx) {
        EXPECT_LE(std::abs(b - a), max_step * 1.1f)
            << "discontinuity at sample " << idx;
    };
    for (int blk = 0; blk < kNumBlocks; ++blk) {
        for (size_t i = 1; i < kAPSineSignalSize; ++i)
            check_step(blocks[blk][i-1], blocks[blk][i],
                       static_cast<size_t>(blk) * kAPSineSignalSize + i);
        if (blk + 1 < kNumBlocks)
            check_step(blocks[blk].back(), blocks[blk+1].front(),
                       static_cast<size_t>(blk + 1) * kAPSineSignalSize);
    }
}

TEST(AniraProcessor, RelativeModelPathLoads) {
    // Verifies that model_path values relative to the JSON config file are resolved correctly.
    EXPECT_NO_THROW(AniraProcessor proc(SINE_OSC_RELATIVE_JSON_PATH));
}

#else  // !USE_LIBTORCH

TEST(AniraProcessor, SkippedWithoutLibTorch) {
    GTEST_SKIP() << "LibTorch not available – AniraProcessor integration test skipped";
}

#endif  // USE_LIBTORCH
