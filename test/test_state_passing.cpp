#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <anira/anira.h>
#include "anira_tilde/state_passing/StatePassingPrePostProcessor.h"

using namespace anira_tilde;

#ifndef STATE_TEST_MODEL_PATH
#error "STATE_TEST_MODEL_PATH must be defined via CMake"
#endif
#ifndef SINE_OSC_MODEL_PATH
#error "SINE_OSC_MODEL_PATH must be defined via CMake"
#endif

#ifdef USE_LIBTORCH

static constexpr size_t kSignalSize  = 128;
static constexpr size_t kStateSize   = 4;
static constexpr int    kTimeoutSecs = 5;

static anira::InferenceConfig make_config() {
    std::vector<anira::ModelData> model_data = {
        {std::string(STATE_TEST_MODEL_PATH), anira::InferenceBackend::LIBTORCH},
    };

    std::vector<anira::TensorShape> tensor_shapes = {
        {
            {{1, 1, static_cast<int64_t>(kSignalSize)},
             {1,    static_cast<int64_t>(kStateSize)}},
            {{1, 1, static_cast<int64_t>(kSignalSize)},
             {1,    static_cast<int64_t>(kStateSize)}},
            anira::InferenceBackend::LIBTORCH,
        },
    };

    anira::ProcessingSpec proc_spec(
        {1, 1},              // preprocess_input_channels
        {1, 1},              // postprocess_output_channels
        {kSignalSize, 0},    // preprocess_input_size  (0 = non-streamable for state)
        {kSignalSize, 0}     // postprocess_output_size
    );

    return {model_data, tensor_shapes, proc_spec, 10.0f};
}

// Wait until inference has completed (available output samples return to prev).
static void wait_for_inference(anira::InferenceHandler& handler, size_t prev) {
    auto deadline = std::chrono::system_clock::now()
                  + std::chrono::seconds(kTimeoutSecs);
    while (handler.get_available_samples(0) != prev) {
        if (std::chrono::system_clock::now() > deadline)
            FAIL() << "Timed out waiting for inference to complete";
        std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    }
}

TEST(StatePassing, StateIsFedBackBetweenInferences) {
    auto config = make_config();
    std::vector<StatePair> state_pairs = {{1, 1}};
    StatePassingPrePostProcessor pp(config, state_pairs);

    // State tensor is zero-initialised by StatePassingPrePostProcessor.
    for (size_t j = 0; j < kStateSize; ++j)
        EXPECT_FLOAT_EQ(pp.get_input(1, j), 0.0f);

    anira::InferenceHandler handler(pp, config);
    handler.prepare({static_cast<float>(kSignalSize), 44100.0f});
    handler.set_inference_backend(anira::InferenceBackend::LIBTORCH);

    std::vector<float> audio(kSignalSize, 0.0f);
    float* ch[1] = {audio.data()};

    // --- First inference ---
    size_t prev = handler.get_available_samples(0);
    handler.process(ch, kSignalSize);
    wait_for_inference(handler, prev);

    // Model adds 1.0 to state; initial state was 0 → expect 1.0.
    for (size_t j = 0; j < kStateSize; ++j)
        EXPECT_FLOAT_EQ(pp.get_input(1, j), 1.0f)
            << "state[" << j << "] wrong after 1st inference";

    // --- Second inference ---
    prev = handler.get_available_samples(0);
    handler.process(ch, kSignalSize);
    wait_for_inference(handler, prev);

    // State was 1.0 going in → expect 2.0 after second inference.
    for (size_t j = 0; j < kStateSize; ++j)
        EXPECT_FLOAT_EQ(pp.get_input(1, j), 2.0f)
            << "state[" << j << "] wrong after 2nd inference";
}

// ---------------------------------------------------------------------------
// Sine oscillator: phase state is passed between inferences
// ---------------------------------------------------------------------------

static constexpr size_t kSineSignalSize  = 512;
static constexpr float  kSampleRate      = 44100.0f;

static anira::InferenceConfig make_sine_config() {
    std::vector<anira::ModelData> model_data = {
        {std::string(SINE_OSC_MODEL_PATH), anira::InferenceBackend::LIBTORCH},
    };

    std::vector<anira::TensorShape> tensor_shapes = {
        {
            // inputs:  freq [1,1,512] (streamable, drives inference), phase [1,1]
            {{1, 1, static_cast<int64_t>(kSineSignalSize)}, {1, 1}},
            // outputs: audio [1,1,512], phase [1,1]
            {{1, 1, static_cast<int64_t>(kSineSignalSize)}, {1, 1}},
            anira::InferenceBackend::LIBTORCH,
        },
    };

    anira::ProcessingSpec proc_spec(
        {1, 1},                      // preprocess_input_channels
        {1, 1},                      // postprocess_output_channels
        {kSineSignalSize, 0},        // preprocess_input_size (freq=512, phase=non-streamable)
        {kSineSignalSize, 0}         // postprocess_output_size (audio=512, phase=non-streamable)
    );

    return {model_data, tensor_shapes, proc_spec, 10.0f};
}

TEST(StatePassing, SineOscillatorPhaseAccumulates) {
    auto config = make_sine_config();
    // output_tensor=1 (phase out) feeds back into input_tensor=1 (phase in)
    std::vector<StatePair> state_pairs = {{1, 1}};
    StatePassingPrePostProcessor pp(config, state_pairs);

    anira::InferenceHandler handler(pp, config);
    handler.prepare({static_cast<float>(kSineSignalSize), kSampleRate});
    handler.set_inference_backend(anira::InferenceBackend::LIBTORCH);

    const float freq = 440.0f;
    std::vector<float> freq_buf(kSineSignalSize, freq);
    std::vector<float> audio_buf(kSineSignalSize, 0.0f);
    const float* freq_ch[1] = {freq_buf.data()};
    float*       out_ch[1]  = {audio_buf.data()};

    const float phase_per_block =
        std::fmod(2.0f * static_cast<float>(M_PI) * freq / kSampleRate
                  * static_cast<float>(kSineSignalSize),
                  2.0f * static_cast<float>(M_PI));

    // --- First inference ---
    size_t prev = handler.get_available_samples(0);
    handler.process(freq_ch, kSineSignalSize, out_ch, kSineSignalSize);
    wait_for_inference(handler, prev);

    const float phase_1 = pp.get_input(1, 0);
    EXPECT_NEAR(phase_1, phase_per_block, 1e-4f)
        << "phase wrong after 1st inference";

    // --- Second inference ---
    prev = handler.get_available_samples(0);
    handler.process(freq_ch, kSineSignalSize, out_ch, kSineSignalSize);
    wait_for_inference(handler, prev);

    const float phase_2 = pp.get_input(1, 0);
    const float expected_phase_2 =
        std::fmod(phase_1 + phase_per_block, 2.0f * static_cast<float>(M_PI));
    EXPECT_NEAR(phase_2, expected_phase_2, 1e-4f)
        << "phase wrong after 2nd inference — state may not have been passed";
}

TEST(StatePassing, SineOscillatorContinuousAcrossBlockBoundaries) {
    auto config = make_sine_config();
    std::vector<StatePair> state_pairs = {{1, 1}};
    StatePassingPrePostProcessor pp(config, state_pairs);

    anira::InferenceHandler handler(pp, config);
    handler.prepare({static_cast<float>(kSineSignalSize), kSampleRate});
    handler.set_inference_backend(anira::InferenceBackend::LIBTORCH);

    const float freq = 440.0f;
    // For a continuous sine, |sample[i] - sample[i-1]| <= 2*pi*f/sr.
    // A discontinuity (state not passed) produces a jump of up to ~2.0.
    const float max_step = 2.0f * static_cast<float>(M_PI) * freq / kSampleRate;

    std::vector<float> freq_buf(kSineSignalSize, freq);
    const float* freq_ch[1] = {freq_buf.data()};

    auto run_block = [&](std::vector<float>& out) {
        out.resize(kSineSignalSize);
        float* out_ch[1] = {out.data()};
        size_t prev = handler.get_available_samples(0);
        handler.process(freq_ch, kSineSignalSize, out_ch, kSineSignalSize);
        wait_for_inference(handler, prev);
    };

    // Discard warm-up blocks to move past the latency-padding zeros.
    std::vector<float> discard;
    int warmup = handler.get_latency() / static_cast<int>(kSineSignalSize) + 1;
    for (int i = 0; i < warmup; ++i)
        run_block(discard);

    // Capture two consecutive real output blocks.
    std::vector<float> block1, block2;
    run_block(block1);
    run_block(block2);

    // Check every consecutive sample pair, including the block boundary.
    auto check = [&](float a, float b, size_t idx) {
        EXPECT_LE(std::abs(b - a), max_step * 1.1f)
            << "discontinuity at sample " << idx;
    };
    for (size_t i = 1; i < kSineSignalSize; ++i)
        check(block1[i-1], block1[i], i);
    check(block1.back(), block2.front(), kSineSignalSize);   // ← the boundary
    for (size_t i = 1; i < kSineSignalSize; ++i)
        check(block2[i-1], block2[i], kSineSignalSize + i);
}


#else  // !USE_LIBTORCH

TEST(StatePassing, SkippedWithoutLibTorch) {
    GTEST_SKIP() << "LibTorch not available – state-passing integration test skipped";
}

#endif  // USE_LIBTORCH
