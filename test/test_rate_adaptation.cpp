#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include "AniraProcessor.h"

#ifndef RATE_ADAPT_JSON_PATH
#error "RATE_ADAPT_JSON_PATH must be defined via CMake"
#endif
#ifndef UPSAMPLE_STATE_JSON_PATH
#error "UPSAMPLE_STATE_JSON_PATH must be defined via CMake"
#endif
#ifndef UPSAMPLE_MULTISTATE_JSON_PATH
#error "UPSAMPLE_MULTISTATE_JSON_PATH must be defined via CMake"
#endif
#ifndef DOWNSAMPLE_JSON_PATH
#error "DOWNSAMPLE_JSON_PATH must be defined via CMake"
#endif

#ifdef USE_LIBTORCH

// Host buffer size used throughout these tests.  output_size=32 is a multiple
// so inference boundaries always fall on callback boundaries (clean timing).
static constexpr size_t kRABuffer     = 16;
static constexpr float  kRASampleRate = 44100.0f;
static constexpr auto   kRACallbackMs = std::chrono::milliseconds(50);

// Model: input_size=1, output_size=32.  One inference fires every 32 output
// samples (= every 2 callbacks at buffer_size=16).
// Model behaviour: output = input.expand(32), so constant input 1.0
// produces 32 samples of 1.0 per inference.

// Input is a ramp [0, 1, 2, ..., kRABuffer-1].
// Correct rate adaptation fires inference at offset 0, so output = [0, 0, ..., 0].
// Without rate adaptation kRABuffer inferences fire per callback; the ring buffer
// overflows (anira logs "Buffer overflow") and keeps only the newest samples,
// producing output ≈ [kRABuffer-1, ...] instead of [0, ...].
TEST(RateAdaptation, UpsampleUsesFirstSampleAtBoundary) {
    AniraProcessor proc(RATE_ADAPT_JSON_PATH);
    proc.prepare(kRABuffer, kRASampleRate);

    std::vector<float> in_buf(kRABuffer);
    for (size_t i = 0; i < kRABuffer; ++i)
        in_buf[i] = static_cast<float>(i);  // [0, 1, 2, ..., 15]
    std::vector<float> out_buf(kRABuffer, 0.0f);

    const float*       in_ch[1]  = { in_buf.data() };
    float*             out_ch[1] = { out_buf.data() };
    const float* const* in_ptrs[1]  = { in_ch };
    float* const*       out_ptrs[1] = { out_ch };
    size_t in_sizes[1]  = { kRABuffer };
    size_t out_sizes[1] = { kRABuffer };

    auto run_callback = [&]() {
        out_buf.assign(kRABuffer, 0.0f);
        proc.process(in_ptrs, in_sizes, out_ptrs, out_sizes);
        std::this_thread::sleep_for(kRACallbackMs);
    };

    // Warm up past latency.
    int warmup = static_cast<int>(proc.get_latency_samples() / kRABuffer) + 2;
    for (int i = 0; i < warmup; ++i)
        run_callback();

    // After warmup every inference uses in_buf[0]=0.0, so output should be 0.0.
    run_callback();

    for (size_t i = 0; i < kRABuffer; ++i)
        EXPECT_FLOAT_EQ(out_buf[i], 0.0f) << "sample " << i;
}

// Verify that the inference boundary falls exactly every output_size samples.
TEST(RateAdaptation, InferenceBoundaryAlignedToOutputSize) {
    AniraProcessor proc(RATE_ADAPT_JSON_PATH);
    proc.prepare(kRABuffer, kRASampleRate);

    std::vector<float> zero_buf(kRABuffer, 0.0f);
    std::vector<float> one_buf(kRABuffer, 1.0f);
    std::vector<float> out_buf(kRABuffer, 0.0f);

    float*             out_ch[1]   = { out_buf.data() };
    float* const*      out_ptrs[1] = { out_ch };
    size_t out_sizes[1] = { kRABuffer };

    auto run_with = [&](const std::vector<float>& in) {
        out_buf.assign(kRABuffer, 0.0f);
        const float*        in_ch[1]   = { in.data() };
        const float* const* in_ptrs[1] = { in_ch };
        size_t in_sizes[1] = { kRABuffer };
        proc.process(in_ptrs, in_sizes, out_ptrs, out_sizes);
        std::this_thread::sleep_for(kRACallbackMs);
    };

    int latency = static_cast<int>(proc.get_latency_samples() / kRABuffer);
    int warmup = latency + 2;
    for (int i = 0; i < warmup; ++i)
        run_with(zero_buf);

    // run one block (32 samples) with ones, then run inferences with zeros until latency has passed, 
    // then check that we get the same as output
    run_with(one_buf);
    run_with(one_buf);

    // Wait for remaining latency to pass after the two inferences above
    for (int i = 0; i < latency - 2; ++i)
        run_with(zero_buf);

    for (size_t i = 0; i < kRABuffer; ++i)
        EXPECT_FLOAT_EQ(out_buf[i], 0.0f) << "sample " << i << " ones appeared too early";

    // The next two runs (16 samples) should return ones - 32x upsampler
    for (int i = 0; i < 2; ++i) {
        run_with(zero_buf);
        
        for (size_t i = 0; i < kRABuffer; ++i)
            EXPECT_FLOAT_EQ(out_buf[i], 1.0f) << "sample " << i << " from one-inference";
    }
    
    run_with(zero_buf);

    for (size_t i = 0; i < kRABuffer; ++i)
        EXPECT_FLOAT_EQ(out_buf[i], 0.0f) << "sample " << i << " should return to zero";
}

// ---------------------------------------------------------------------------
// Downsampler: input_size=32, output_size=1 (N=32)
// Model: mean of 32 input samples → 1 output sample.
// Design spec: that 1 output sample is sample-and-held across all
// output positions until the next inference fires.
// ---------------------------------------------------------------------------

// kRABuffer=16 is also used here: input_size=32 == 2*kRABuffer, so
// inference boundaries fall exactly every two callbacks.

// Constant input 2.0 should produce 2.0 across the entire output buffer
// once the pipeline is warm.  Requires sample-and-hold (design spec).
TEST(RateAdaptation, DownsampleHoldsConstantValue) {
#ifdef ANIRA_TILDE_WITH_ASAN
    GTEST_SKIP() << "Skipped under AddressSanitizer: instrumentation slows "
                 << "inference past max_inference_time, anira drops output.";
#endif
    AniraProcessor proc(DOWNSAMPLE_JSON_PATH);
    proc.prepare(kRABuffer, kRASampleRate);

    std::vector<float> in_buf(kRABuffer, 2.0f);
    std::vector<float> out_buf(kRABuffer, 0.0f);

    const float*        in_ch[1]    = { in_buf.data() };
    float*              out_ch[1]   = { out_buf.data() };
    const float* const* in_ptrs[1]  = { in_ch };
    float* const*       out_ptrs[1] = { out_ch };
    size_t in_sizes[1]  = { kRABuffer };
    size_t out_sizes[1] = { kRABuffer };

    auto run_callback = [&]() {
        out_buf.assign(kRABuffer, 0.0f);
        proc.process(in_ptrs, in_sizes, out_ptrs, out_sizes);
        std::this_thread::sleep_for(kRACallbackMs);
    };

    int warmup = static_cast<int>(proc.get_latency_samples() / kRABuffer) + 2;
    for (int i = 0; i < warmup; ++i)
        run_callback();

    run_callback();

    for (size_t i = 0; i < kRABuffer; ++i)
        EXPECT_FLOAT_EQ(out_buf[i], 2.0f) << "sample " << i;
}

// Inference boundary falls exactly every input_size=32 samples (= 2 callbacks).
// After zeros are stable, switching to 2.0 for one full boundary window should
// flip all output to 2.0.
TEST(RateAdaptation, DownsampleInferenceBoundaryAlignedToInputSize) {
    AniraProcessor proc(DOWNSAMPLE_JSON_PATH);
    proc.prepare(kRABuffer, kRASampleRate);

    std::vector<float> zero_buf(kRABuffer, 0.0f);
    std::vector<float> two_buf(kRABuffer, 2.0f);
    std::vector<float> out_buf(kRABuffer, 0.0f);

    float*             out_ch[1]    = { out_buf.data() };
    float* const*      out_ptrs[1]  = { out_ch };
    size_t out_sizes[1] = { kRABuffer };

    auto run_with = [&](const std::vector<float>& in) {
        out_buf.assign(kRABuffer, 0.0f);
        const float*        in_ch[1]   = { in.data() };
        const float* const* in_ptrs[1] = { in_ch };
        size_t in_sizes[1] = { kRABuffer };
        proc.process(in_ptrs, in_sizes, out_ptrs, out_sizes);
        std::this_thread::sleep_for(kRACallbackMs);
    };

    int warmup = static_cast<int>(proc.get_latency_samples() / kRABuffer) + 2;
    for (int i = 0; i < warmup; ++i)
        run_with(zero_buf);

    // input_size=32 == 2*kRABuffer: two callbacks = one inference boundary.
    // Drain two full boundaries of zeros so the ring buffer is empty.
    run_with(zero_buf);  // callback 1/2 of first zero boundary
    run_with(zero_buf);  // callback 2/2 → zero-inference fires, pops zeros
    run_with(zero_buf);  // callback 1/2 of second zero boundary
    run_with(zero_buf);  // callback 2/2 → zero-inference, ring buffer drained

    // Push exactly one inference boundary of 2.0.
    run_with(two_buf);   // callback 1/2 — 16 samples accumulated
    run_with(two_buf);   // callback 2/2 — 32 total → two-inference fires (async)
    run_with(two_buf);   // callback 3 — 50ms sleep has elapsed, result is in ring, pop it

    // out_buf should contain the 2.0 inference result (sample-and-hold).
    for (size_t i = 0; i < kRABuffer; ++i)
        EXPECT_FLOAT_EQ(out_buf[i], 2.0f) << "sample " << i;
}

// ---------------------------------------------------------------------------
// Combined: rate adaptation + state passing
// upsample_state model: input_size=1, output_size=16, plus a state tensor
// that accumulates (state_out = state_in + 1.0 per inference).
// audio = latent + state_before_inference (so with latent=0, audio == state).
// ---------------------------------------------------------------------------

// With latent=0 the audio output equals the pre-inference state value, which
// increments by 1 each inference.  Two consecutive output blocks should:
//   (a) each have all-equal samples (one inference fills one block of 16), and
//   (b) differ by a positive integer multiple of 1.0 (state is advancing).
TEST(RateAdaptation, UpsampleWithStateRateAdaptationAndStatePassedForward) {
    AniraProcessor proc(UPSAMPLE_STATE_JSON_PATH);
    proc.prepare(kRABuffer, kRASampleRate);

    std::vector<float> in_buf(kRABuffer, 0.0f);  // latent = 0 → audio = state
    std::vector<float> out_buf(kRABuffer, 0.0f);

    const float*        in_ch[1]    = { in_buf.data() };
    float*              out_ch[1]   = { out_buf.data() };
    const float* const* in_ptrs[1]  = { in_ch };
    float* const*       out_ptrs[1] = { out_ch };
    size_t in_sizes[1]  = { kRABuffer };
    size_t out_sizes[1] = { kRABuffer };

    auto run_callback = [&]() {
        out_buf.assign(kRABuffer, 0.0f);
        proc.process(in_ptrs, in_sizes, out_ptrs, out_sizes);
        std::this_thread::sleep_for(kRACallbackMs);
    };

    int warmup = static_cast<int>(proc.get_latency_samples() / kRABuffer) + 2;
    for (int i = 0; i < warmup; ++i)
        run_callback();

    // Capture two consecutive post-warmup blocks.
    run_callback();
    const float v1 = out_buf[0];
    for (size_t i = 1; i < kRABuffer; ++i)
        EXPECT_FLOAT_EQ(out_buf[i], v1) << "block1 sample " << i << " not uniform (one inference should fill one block)";

    run_callback();
    const float v2 = out_buf[0];
    for (size_t i = 1; i < kRABuffer; ++i)
        EXPECT_FLOAT_EQ(out_buf[i], v2) << "block2 sample " << i << " not uniform";

    // Each pop drains exactly one inference worth of output (output_size=16 samples).
    // Consecutive inferences are separated by exactly 1 state increment, so delta == 1.
    EXPECT_FLOAT_EQ(v2 - v1, 1.0f) << "expected state delta of 1 between consecutive blocks; got " << (v2 - v1);
}

// ---------------------------------------------------------------------------
// Upsample + MULTIPLE state tensors
// Reproduces the crash where adj_out arrays were sized to n_sig_out (1) but
// InferenceManager iterates all output tensors (3: 1 signal + 2 state).
// The out-of-bounds read of num_samples[1] / num_samples[2] returns garbage;
// with two state tensors this reliably hits non-zero memory and causes either
// a SIGSEGV or memory corruption detectable via incorrect audio output.
// ---------------------------------------------------------------------------
TEST(RateAdaptation, UpsampleWithMultipleStateTensorsDoesNotCrash) {
    AniraProcessor proc(UPSAMPLE_MULTISTATE_JSON_PATH);
    proc.prepare(kRABuffer, kRASampleRate);

    std::vector<float> in_buf(kRABuffer, 0.0f);
    std::vector<float> out_buf(kRABuffer, 0.0f);

    const float*        in_ch[1]    = { in_buf.data() };
    float*              out_ch[1]   = { out_buf.data() };
    const float* const* in_ptrs[1]  = { in_ch };
    float* const*       out_ptrs[1] = { out_ch };
    size_t in_sizes[1]  = { kRABuffer };
    size_t out_sizes[1] = { kRABuffer };

    auto run_callback = [&]() {
        out_buf.assign(kRABuffer, 0.0f);
        proc.process(in_ptrs, in_sizes, out_ptrs, out_sizes);
        std::this_thread::sleep_for(kRACallbackMs);
    };

    int warmup = static_cast<int>(proc.get_latency_samples() / kRABuffer) + 2;
    for (int i = 0; i < warmup; ++i)
        run_callback();

    run_callback();
    const float v1 = out_buf[0];
    for (size_t i = 1; i < kRABuffer; ++i)
        EXPECT_FLOAT_EQ(out_buf[i], v1) << "block1 sample " << i << " not uniform";

    run_callback();
    const float v2 = out_buf[0];
    for (size_t i = 1; i < kRABuffer; ++i)
        EXPECT_FLOAT_EQ(out_buf[i], v2) << "block2 sample " << i << " not uniform";

    EXPECT_FLOAT_EQ(v2 - v1, 1.0f)
        << "expected state delta of 1 between consecutive blocks; got " << (v2 - v1);
}

#else  // !USE_LIBTORCH

TEST(RateAdaptation, SkippedWithoutLibTorch) {
    GTEST_SKIP() << "LibTorch not available – rate-adaptation integration test skipped";
}

#endif  // USE_LIBTORCH
