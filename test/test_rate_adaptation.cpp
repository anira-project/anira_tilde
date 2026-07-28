#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <thread>
#include <vector>

#include "TestBackends.h"
#include "anira_tilde/inference/Session.h"
#include "anira_tilde/inference/TensorLayout.h"

using namespace anira_tilde;

using anira_tilde_test::Backend;

// Runs once per compiled-in backend (see TestBackends.h); LibTorch coverage
// exists only when the build enables ANIRA_WITH_LIBTORCH.
class RateAdaptation : public testing::TestWithParam<Backend> {};

// Host buffer size used throughout these tests.  output_size=32 is a multiple
// so inference boundaries always fall on callback boundaries (clean timing).
static constexpr size_t k_ra_buffer = 16;
static constexpr size_t k_ra_output_size = 32;  // upsampler_x32 model output block
static constexpr float k_ra_sample_rate = 44100.0f;
static constexpr auto k_ra_callback_ms = std::chrono::milliseconds(50);

// Per-build extra warmup callbacks. Slow build flavours (sanitizers, x86_64
// under Rosetta) bake a larger max_inference_time into the JSON via
// ANIRA_TILDE_TEST_MAX_INFERENCE_TIME_OVERHEAD; the test then has to wait
// at least one inference budget after latency drains, otherwise a single
// in-flight inference can still be unfinished when we read the output.
#ifndef ANIRA_TILDE_TEST_MAX_INFERENCE_TIME_OVERHEAD_MS
#define ANIRA_TILDE_TEST_MAX_INFERENCE_TIME_OVERHEAD_MS 0
#endif
static constexpr int k_ra_extra_warmup_callbacks =
    2 + (ANIRA_TILDE_TEST_MAX_INFERENCE_TIME_OVERHEAD_MS + 49) / 50;

// Model: input_size=1, output_size=32.  One inference fires every 32 output
// samples (= every 2 callbacks at buffer_size=16).
// Model behaviour: output = input.expand(32), so constant input 1.0
// produces 32 samples of 1.0 per inference.

// Input is a ramp [0, 1, 2, ..., kRABuffer-1].
// Correct rate adaptation fires inference at offset 0, so output = [0, 0, ..., 0].
// Without rate adaptation kRABuffer inferences fire per callback; the ring buffer
// overflows (anira logs "Buffer overflow") and keeps only the newest samples,
// producing output ≈ [kRABuffer-1, ...] instead of [0, ...].
TEST_P(RateAdaptation, UpsampleUsesFirstSampleAtBoundary) {
    anira_tilde::Session proc(anira_tilde_test::json_path("rate_adapt_test", GetParam()));
    proc.prepare(k_ra_buffer, k_ra_sample_rate);

    std::vector<float> in_buf(k_ra_buffer);
    for (size_t i = 0; i < k_ra_buffer; ++i) {
        in_buf[i] = static_cast<float>(i);  // [0, 1, 2, ..., 15]
    }
    std::vector<float> out_buf(k_ra_buffer, 0.0f);

    std::array<const float*, 1> in_ch = {in_buf.data()};
    std::array<float*, 1> out_ch = {out_buf.data()};
    std::array<const float* const*, 1> in_ptrs = {in_ch.data()};
    std::array<float* const*, 1> out_ptrs = {out_ch.data()};
    std::array<size_t, 1> in_sizes = {k_ra_buffer};
    std::array<size_t, 1> out_sizes = {k_ra_buffer};

    auto run_callback = [&]() {
        out_buf.assign(k_ra_buffer, 0.0f);
        proc.process(in_ptrs.data(), in_sizes.data(), out_ptrs.data(), out_sizes.data());
        std::this_thread::sleep_for(k_ra_callback_ms);
    };

    // Warm up past latency.
    int const warmup =
        static_cast<int>(proc.get_latency_samples() / k_ra_buffer) + k_ra_extra_warmup_callbacks;
    for (int i = 0; i < warmup; ++i) { run_callback(); }

    // After warmup every inference uses in_buf[0]=0.0, so output should be 0.0.
    run_callback();

    for (size_t i = 0; i < k_ra_buffer; ++i) {
        EXPECT_FLOAT_EQ(out_buf[i], 0.0f) << "sample " << i;
    }
}

// Verify that the inference boundary falls exactly every output_size samples.
TEST_P(RateAdaptation, InferenceBoundaryAlignedToOutputSize) {
    anira_tilde::Session proc(anira_tilde_test::json_path("rate_adapt_test", GetParam()));
    proc.prepare(k_ra_buffer, k_ra_sample_rate);

    std::vector<float> const zero_buf(k_ra_buffer, 0.0f);
    std::vector<float> const one_buf(k_ra_buffer, 1.0f);
    std::vector<float> out_buf(k_ra_buffer, 0.0f);

    std::array<float*, 1> out_ch = {out_buf.data()};
    std::array<float* const*, 1> out_ptrs = {out_ch.data()};
    std::array<size_t, 1> out_sizes = {k_ra_buffer};

    auto run_with = [&](const std::vector<float>& in) {
        out_buf.assign(k_ra_buffer, 0.0f);
        std::array<const float*, 1> in_ch = {in.data()};
        std::array<const float* const*, 1> in_ptrs = {in_ch.data()};
        std::array<size_t, 1> in_sizes = {k_ra_buffer};
        proc.process(in_ptrs.data(), in_sizes.data(), out_ptrs.data(), out_sizes.data());
        std::this_thread::sleep_for(k_ra_callback_ms);
    };

    int const latency = static_cast<int>(proc.get_latency_samples() / k_ra_buffer);
    int const warmup = latency + 2;
    for (int i = 0; i < warmup; ++i) { run_with(zero_buf); }

    // run one block (32 samples) with ones, then run inferences with zeros until latency has
    // passed, then check that we get the same as output
    run_with(one_buf);
    run_with(one_buf);

    // Wait for remaining latency to pass after the two inferences above
    for (int i = 0; i < latency - 2; ++i) { run_with(zero_buf); }

    for (size_t i = 0; i < k_ra_buffer; ++i) {
        EXPECT_FLOAT_EQ(out_buf[i], 0.0f) << "sample " << i << " ones appeared too early";
    }

    // The next two runs (16 samples) should return ones - 32x upsampler
    for (int i = 0; i < 2; ++i) {
        run_with(zero_buf);

        for (size_t i = 0; i < k_ra_buffer; ++i) {
            EXPECT_FLOAT_EQ(out_buf[i], 1.0f) << "sample " << i << " from one-inference";
        }
    }

    run_with(zero_buf);

    for (size_t i = 0; i < k_ra_buffer; ++i) {
        EXPECT_FLOAT_EQ(out_buf[i], 0.0f) << "sample " << i << " should return to zero";
    }
}

// Host buffer LARGER than the model output block (64 = 2×32): two inferences
// must fire per callback, one per output-block boundary. This is the RAVE
// decoder regime (host 2048, output block 128) — a single-fire adaptor
// starves anira and the output collapses to silence.
TEST_P(RateAdaptation, UpsampleFiresOncePerOutputBlockBoundary) {
    constexpr size_t k_big = 2 * k_ra_output_size;  // 64 = two output blocks
    anira_tilde::Session proc(anira_tilde_test::json_path("rate_adapt_test", GetParam()));
    proc.prepare(k_big, k_ra_sample_rate);

    // Boundaries fall at offsets 0 and 32 of every callback, so inferences
    // alternate between in_buf[0]=1 and in_buf[32]=33 (never 0 — a zero in
    // the steady-state output would mean a starved/underrun ring instead).
    std::vector<float> in_buf(k_big);
    for (size_t i = 0; i < k_big; ++i) { in_buf[i] = static_cast<float>(i + 1); }
    std::vector<float> out_buf(k_big, 0.0f);

    std::array<const float*, 1> in_ch = {in_buf.data()};
    std::array<float*, 1> out_ch = {out_buf.data()};
    std::array<const float* const*, 1> in_ptrs = {in_ch.data()};
    std::array<float* const*, 1> out_ptrs = {out_ch.data()};
    std::array<size_t, 1> in_sizes = {k_big};
    std::array<size_t, 1> out_sizes = {k_big};

    auto run_callback = [&]() {
        out_buf.assign(k_big, 0.0f);
        proc.process(in_ptrs.data(), in_sizes.data(), out_ptrs.data(), out_sizes.data());
        std::this_thread::sleep_for(k_ra_callback_ms);
    };

    int const warmup =
        static_cast<int>(proc.get_latency_samples() / k_big) + k_ra_extra_warmup_callbacks;
    for (int i = 0; i < warmup; ++i) { run_callback(); }

    // Collect four callbacks of steady-state output.
    std::vector<float> collected;
    for (int cb = 0; cb < 4; ++cb) {
        run_callback();
        collected.insert(collected.end(), out_buf.begin(), out_buf.end());
    }

    // Both boundary values must appear, in equal amounts, in runs of exactly
    // one output block (32 samples) — phase depends on latency, structure not.
    size_t ones = 0;
    for (size_t i = 0; i < collected.size(); ++i) {
        ASSERT_TRUE(collected[i] == 1.0f || collected[i] == 33.0f)
            << "sample " << i << " = " << collected[i];
        if (collected[i] == 1.0f) { ++ones; }
    }
    EXPECT_EQ(ones, collected.size() / 2);
    for (size_t i = 1; i < collected.size(); ++i) {
        if (collected[i] == collected[i - 1]) { continue; }
        for (size_t j = i; j < std::min(i + k_ra_output_size, collected.size()); ++j) {
            EXPECT_EQ(collected[j], collected[i]) << "run broken at sample " << j;
        }
        i += k_ra_output_size - 1;
    }
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
TEST_P(RateAdaptation, DownsampleHoldsConstantValue) {
    anira_tilde::Session proc(anira_tilde_test::json_path("downsample_test", GetParam()));
    proc.prepare(k_ra_buffer, k_ra_sample_rate);

    std::vector<float> in_buf(k_ra_buffer, 2.0f);
    std::vector<float> out_buf(k_ra_buffer, 0.0f);

    std::array<const float*, 1> in_ch = {in_buf.data()};
    std::array<float*, 1> out_ch = {out_buf.data()};
    std::array<const float* const*, 1> in_ptrs = {in_ch.data()};
    std::array<float* const*, 1> out_ptrs = {out_ch.data()};
    std::array<size_t, 1> in_sizes = {k_ra_buffer};
    std::array<size_t, 1> out_sizes = {k_ra_buffer};

    auto run_callback = [&]() {
        out_buf.assign(k_ra_buffer, 0.0f);
        proc.process(in_ptrs.data(), in_sizes.data(), out_ptrs.data(), out_sizes.data());
        std::this_thread::sleep_for(k_ra_callback_ms);
    };

    int const warmup =
        static_cast<int>(proc.get_latency_samples() / k_ra_buffer) + k_ra_extra_warmup_callbacks;
    for (int i = 0; i < warmup; ++i) { run_callback(); }

    run_callback();

    for (size_t i = 0; i < k_ra_buffer; ++i) {
        EXPECT_FLOAT_EQ(out_buf[i], 2.0f) << "sample " << i;
    }
}

// Inference boundary falls exactly every input_size=32 samples (= 2 callbacks).
// After zeros are stable, switching to 2.0 for one full boundary window should
// flip all output to 2.0.
TEST_P(RateAdaptation, DownsampleInferenceBoundaryAlignedToInputSize) {
    anira_tilde::Session proc(anira_tilde_test::json_path("downsample_test", GetParam()));
    proc.prepare(k_ra_buffer, k_ra_sample_rate);

    std::vector<float> const zero_buf(k_ra_buffer, 0.0f);
    std::vector<float> const two_buf(k_ra_buffer, 2.0f);
    std::vector<float> out_buf(k_ra_buffer, 0.0f);

    std::array<float*, 1> out_ch = {out_buf.data()};
    std::array<float* const*, 1> out_ptrs = {out_ch.data()};
    std::array<size_t, 1> out_sizes = {k_ra_buffer};

    auto run_with = [&](const std::vector<float>& in) {
        out_buf.assign(k_ra_buffer, 0.0f);
        std::array<const float*, 1> in_ch = {in.data()};
        std::array<const float* const*, 1> in_ptrs = {in_ch.data()};
        std::array<size_t, 1> in_sizes = {k_ra_buffer};
        proc.process(in_ptrs.data(), in_sizes.data(), out_ptrs.data(), out_sizes.data());
        std::this_thread::sleep_for(k_ra_callback_ms);
    };

    int const warmup =
        static_cast<int>(proc.get_latency_samples() / k_ra_buffer) + k_ra_extra_warmup_callbacks;
    for (int i = 0; i < warmup; ++i) { run_with(zero_buf); }

    // input_size=32 == 2*kRABuffer: two callbacks = one inference boundary.
    // Drain two full boundaries of zeros so the ring buffer is empty.
    run_with(zero_buf);  // callback 1/2 of first zero boundary
    run_with(zero_buf);  // callback 2/2 → zero-inference fires, pops zeros
    run_with(zero_buf);  // callback 1/2 of second zero boundary
    run_with(zero_buf);  // callback 2/2 → zero-inference, ring buffer drained

    // Push exactly one inference boundary of 2.0.
    run_with(two_buf);  // callback 1/2 — 16 samples accumulated
    run_with(two_buf);  // callback 2/2 — 32 total → two-inference fires (async)
    run_with(two_buf);  // callback 3 — 50ms sleep has elapsed, result is in ring, pop it

    // out_buf should contain the 2.0 inference result (sample-and-hold).
    for (size_t i = 0; i < k_ra_buffer; ++i) {
        EXPECT_FLOAT_EQ(out_buf[i], 2.0f) << "sample " << i;
    }
}

// Host buffer LARGER than the model input block (64 = 2×32): two inferences
// complete per callback and BOTH results must reach the output, each held
// across its own 32-sample boundary segment. This is the RAVE encoder regime
// (host 2048, input block 128) — a single-pop adaptor drops 15 of every 16
// latents through ring overflow and holds one stale value across the block.
TEST_P(RateAdaptation, DownsamplePopsOncePerInputBlockBoundary) {
    constexpr size_t k_big = 64;  // two input blocks of the mean model
    constexpr size_t k_in_block = 32;
    anira_tilde::Session proc(anira_tilde_test::json_path("downsample_test", GetParam()));
    proc.prepare(k_big, k_ra_sample_rate);

    // First input block all 1.0, second all 3.0 → the mean model pops
    // alternating 1.0 / 3.0, one value per 32-sample segment.
    std::vector<float> in_buf(k_big);
    for (size_t i = 0; i < k_big; ++i) { in_buf[i] = i < k_in_block ? 1.0f : 3.0f; }
    std::vector<float> out_buf(k_big, 0.0f);

    std::array<const float*, 1> in_ch = {in_buf.data()};
    std::array<float*, 1> out_ch = {out_buf.data()};
    std::array<const float* const*, 1> in_ptrs = {in_ch.data()};
    std::array<float* const*, 1> out_ptrs = {out_ch.data()};
    std::array<size_t, 1> in_sizes = {k_big};
    std::array<size_t, 1> out_sizes = {k_big};

    auto run_callback = [&]() {
        out_buf.assign(k_big, 0.0f);
        proc.process(in_ptrs.data(), in_sizes.data(), out_ptrs.data(), out_sizes.data());
        std::this_thread::sleep_for(k_ra_callback_ms);
    };

    // The reported latency counts samples of the OUTPUT tensor's stream,
    // which runs at 1/kInBlock of the host rate here.
    int const warmup = static_cast<int>(proc.get_latency_samples() * k_in_block / k_big) +
                       k_ra_extra_warmup_callbacks;
    for (int i = 0; i < warmup; ++i) { run_callback(); }

    std::vector<float> collected;
    for (int cb = 0; cb < 4; ++cb) {
        run_callback();
        collected.insert(collected.end(), out_buf.begin(), out_buf.end());
    }

    size_t ones = 0;
    for (size_t i = 0; i < collected.size(); ++i) {
        ASSERT_TRUE(collected[i] == 1.0f || collected[i] == 3.0f)
            << "sample " << i << " = " << collected[i];
        if (collected[i] == 1.0f) { ++ones; }
    }
    EXPECT_EQ(ones, collected.size() / 2);
    for (size_t i = 1; i < collected.size(); ++i) {
        if (collected[i] == collected[i - 1]) { continue; }
        for (size_t j = i; j < std::min(i + k_in_block, collected.size()); ++j) {
            EXPECT_EQ(collected[j], collected[i]) << "run broken at sample " << j;
        }
        i += k_in_block - 1;
    }
}

// ---------------------------------------------------------------------------
// Multi-frame blocks (unit level, no backend): a model block can carry
// several frames of a slower stream — e.g. 8 latent frames per 1024-sample
// block, one frame per 128 samples of host time. The adaptor must gather
// input frames at that stride (not consecutively) and play popped output
// frames each across its own sub-segment.
// ---------------------------------------------------------------------------

#include "anira_tilde/rate_adaptation/RateAdaptor.h"

namespace {

anira_tilde::TensorLayout make_pair_layout(size_t in_block, size_t out_block) {
    anira_tilde::TensorLayout layout;
    layout.m_sig_input_channels = {1};
    layout.m_sig_output_channels = {1};
    layout.m_input_block_sizes = {in_block};
    layout.m_output_block_sizes = {out_block};
    return layout;
}

}  // namespace

// Upsample pair, 4 frames per 16-sample block: each gathered frame j must
// come from host offset boundary + j*4, not boundary + j.
TEST(RateAdaptorUnit, UpsampleGathersFramesAtStride) {
    auto layout = make_pair_layout(/*in_block=*/4, /*out_block=*/16);
    anira_tilde::RateAdaptor adaptor;
    adaptor.prepare(layout, /*max_block_size=*/32);

    std::vector<float> in(32), out(32, 0.0f);
    for (size_t i = 0; i < in.size(); ++i) { in[i] = static_cast<float>(i); }

    std::array<const float*, 1> in_ch = {in.data()};
    std::array<float*, 1> out_ch = {out.data()};
    std::array<const float* const*, 1> in_ptrs = {in_ch.data()};
    std::array<float* const*, 1> out_ptrs = {out_ch.data()};
    std::array<size_t, 1> in_sizes = {32};
    std::array<size_t, 1> out_sizes = {32};

    auto view = adaptor.pre_dispatch(layout,
                                     in_ptrs.data(),
                                     in_sizes.data(),
                                     out_ptrs.data(),
                                     out_sizes.data());

    // Two boundaries (offsets 0 and 16) → two gathered blocks of 4 frames.
    ASSERT_EQ(view.m_in_sample_counts[0], 8u);
    const float* gathered = view.m_in_tensors[0][0];
    const std::array<float, 8> expected = {0, 4, 8, 12, 16, 20, 24, 28};
    for (size_t j = 0; j < 8; ++j) { EXPECT_FLOAT_EQ(gathered[j], expected[j]) << "frame " << j; }
}

// Downsample pair, 4 frames per 16-sample input block: each popped frame
// must be held across its own 4-sample sub-segment, in order.
TEST(RateAdaptorUnit, DownsampleHoldsEachFrameAcrossItsSubSegment) {
    auto layout = make_pair_layout(/*in_block=*/16, /*out_block=*/4);
    anira_tilde::RateAdaptor adaptor;
    adaptor.prepare(layout, /*max_block_size=*/32);

    std::vector<float> in(32, 0.0f), out(32, -1.0f);
    std::array<const float*, 1> in_ch = {in.data()};
    std::array<float*, 1> out_ch = {out.data()};
    std::array<const float* const*, 1> in_ptrs = {in_ch.data()};
    std::array<float* const*, 1> out_ptrs = {out_ch.data()};
    std::array<size_t, 1> in_sizes = {32};
    std::array<size_t, 1> out_sizes = {32};

    auto view = adaptor.pre_dispatch(layout,
                                     in_ptrs.data(),
                                     in_sizes.data(),
                                     out_ptrs.data(),
                                     out_sizes.data());

    // Two boundaries (offsets 0 and 16) → two block pops of 4 frames each;
    // play the role of anira and write 1..8 into the pop scratch.
    ASSERT_EQ(view.m_out_sample_counts[0], 8u);
    for (size_t j = 0; j < 8; ++j) { view.m_out_tensors[0][0][j] = static_cast<float>(j + 1); }

    adaptor.post_dispatch(layout, out_ptrs.data(), out_sizes.data());

    for (size_t s = 0; s < 32; ++s) {
        const size_t sub_segment = s / 4;
        EXPECT_FLOAT_EQ(out[s], static_cast<float>(sub_segment + 1)) << "sample " << s;
    }
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
TEST_P(RateAdaptation, UpsampleWithStateRateAdaptationAndStatePassedForward) {
    anira_tilde::Session proc(anira_tilde_test::json_path("upsample_state_test", GetParam()));
    proc.prepare(k_ra_buffer, k_ra_sample_rate);

    std::vector<float> in_buf(k_ra_buffer, 0.0f);  // latent = 0 → audio = state
    std::vector<float> out_buf(k_ra_buffer, 0.0f);

    std::array<const float*, 1> in_ch = {in_buf.data()};
    std::array<float*, 1> out_ch = {out_buf.data()};
    std::array<const float* const*, 1> in_ptrs = {in_ch.data()};
    std::array<float* const*, 1> out_ptrs = {out_ch.data()};
    std::array<size_t, 1> in_sizes = {k_ra_buffer};
    std::array<size_t, 1> out_sizes = {k_ra_buffer};

    auto run_callback = [&]() {
        out_buf.assign(k_ra_buffer, 0.0f);
        proc.process(in_ptrs.data(), in_sizes.data(), out_ptrs.data(), out_sizes.data());
        std::this_thread::sleep_for(k_ra_callback_ms);
    };

    int const warmup =
        static_cast<int>(proc.get_latency_samples() / k_ra_buffer) + k_ra_extra_warmup_callbacks;
    for (int i = 0; i < warmup; ++i) { run_callback(); }

    // Capture two consecutive post-warmup blocks.
    run_callback();
    const float v1 = out_buf[0];
    for (size_t i = 1; i < k_ra_buffer; ++i) {
        EXPECT_FLOAT_EQ(out_buf[i], v1)
            << "block1 sample " << i << " not uniform (one inference should fill one block)";
    }

    run_callback();
    const float v2 = out_buf[0];
    for (size_t i = 1; i < k_ra_buffer; ++i) {
        EXPECT_FLOAT_EQ(out_buf[i], v2) << "block2 sample " << i << " not uniform";
    }

    // Each pop drains exactly one inference worth of output (output_size=16 samples).
    // Consecutive inferences are separated by exactly 1 state increment, so delta == 1.
    EXPECT_FLOAT_EQ(v2 - v1, 1.0f)
        << "expected state delta of 1 between consecutive blocks; got " << (v2 - v1);
}

// ---------------------------------------------------------------------------
// Upsample + MULTIPLE state tensors
// Reproduces the crash where adj_out arrays were sized to n_sig_out (1) but
// InferenceManager iterates all output tensors (3: 1 signal + 2 state).
// The out-of-bounds read of num_samples[1] / num_samples[2] returns garbage;
// with two state tensors this reliably hits non-zero memory and causes either
// a SIGSEGV or memory corruption detectable via incorrect audio output.
// ---------------------------------------------------------------------------
TEST_P(RateAdaptation, UpsampleWithMultipleStateTensorsDoesNotCrash) {
    anira_tilde::Session proc(anira_tilde_test::json_path("upsample_multistate_test", GetParam()));
    proc.prepare(k_ra_buffer, k_ra_sample_rate);

    std::vector<float> in_buf(k_ra_buffer, 0.0f);
    std::vector<float> out_buf(k_ra_buffer, 0.0f);

    std::array<const float*, 1> in_ch = {in_buf.data()};
    std::array<float*, 1> out_ch = {out_buf.data()};
    std::array<const float* const*, 1> in_ptrs = {in_ch.data()};
    std::array<float* const*, 1> out_ptrs = {out_ch.data()};
    std::array<size_t, 1> in_sizes = {k_ra_buffer};
    std::array<size_t, 1> out_sizes = {k_ra_buffer};

    auto run_callback = [&]() {
        out_buf.assign(k_ra_buffer, 0.0f);
        proc.process(in_ptrs.data(), in_sizes.data(), out_ptrs.data(), out_sizes.data());
        std::this_thread::sleep_for(k_ra_callback_ms);
    };

    int const warmup =
        static_cast<int>(proc.get_latency_samples() / k_ra_buffer) + k_ra_extra_warmup_callbacks;
    for (int i = 0; i < warmup; ++i) { run_callback(); }

    run_callback();
    const float v1 = out_buf[0];
    for (size_t i = 1; i < k_ra_buffer; ++i) {
        EXPECT_FLOAT_EQ(out_buf[i], v1) << "block1 sample " << i << " not uniform";
    }

    run_callback();
    const float v2 = out_buf[0];
    for (size_t i = 1; i < k_ra_buffer; ++i) {
        EXPECT_FLOAT_EQ(out_buf[i], v2) << "block2 sample " << i << " not uniform";
    }

    EXPECT_FLOAT_EQ(v2 - v1, 1.0f)
        << "expected state delta of 1 between consecutive blocks; got " << (v2 - v1);
}

INSTANTIATE_TEST_SUITE_P(Backends,
                         RateAdaptation,
                         testing::ValuesIn(anira_tilde_test::backends()),
                         anira_tilde_test::param_name);
