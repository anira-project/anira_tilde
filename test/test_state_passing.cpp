#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numbers>
#include <thread>
#include <vector>

#include "TestBackends.h"
#include "anira/InferenceConfig.h"
#include "anira/InferenceHandler.h"
#include "anira_tilde/state_passing/StatePairParser.h"
#include "anira_tilde/state_passing/StatePassingPrePostProcessor.h"

using namespace anira_tilde;
using anira_tilde_test::Backend;

// Runs once per compiled-in backend (see TestBackends.h); LibTorch coverage
// exists only when the build enables ANIRA_WITH_LIBTORCH.
class StatePassing : public testing::TestWithParam<Backend> {};

static constexpr size_t k_signal_size = 128;
static constexpr size_t k_state_size = 4;
static constexpr int k_timeout_secs = 5;

namespace {

anira::InferenceConfig make_config(const Backend& b) {
    std::vector<anira::ModelData> const model_data = {
        {anira_tilde_test::model_path("state_accumulator", b), b.backend},
    };

    // Universal shape (no backend tag): valid for whichever backend runs.
    std::vector<anira::TensorShape> const tensor_shapes = {
        {
            {{1, 1, static_cast<int64_t>(k_signal_size)}, {1, static_cast<int64_t>(k_state_size)}},
            {{1, 1, static_cast<int64_t>(k_signal_size)}, {1, static_cast<int64_t>(k_state_size)}},
        },
    };

    anira::ProcessingSpec const proc_spec({1, 1},              // preprocess_input_channels
                                          {1, 1},              // postprocess_output_channels
                                          {k_signal_size, 0},  // preprocess_input_size  (0 =
                                                               // non-streamable for state)
                                          {k_signal_size, 0}   // postprocess_output_size
    );

    return {model_data, tensor_shapes, proc_spec, 10.0f};
}

// Wait until inference has completed (available output samples return to prev).
void wait_for_inference(anira::InferenceHandler& handler, size_t prev) {
    auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(k_timeout_secs);
    while (handler.get_available_samples(0) != prev) {
        if (std::chrono::system_clock::now() > deadline) {
            FAIL() << "Timed out waiting for inference to complete";
        }
        std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    }
}

}  // namespace

TEST_P(StatePassing, StateIsFedBackBetweenInferences) {
    auto config = make_config(GetParam());
    std::vector<StatePair> const state_pairs = {{.m_output_tensor = 1, .m_input_tensor = 1}};
    StatePassingPrePostProcessor pp(config, state_pairs);

    // State tensor is zero-initialised by StatePassingPrePostProcessor.
    for (size_t j = 0; j < k_state_size; ++j) { EXPECT_FLOAT_EQ(pp.get_input(1, j), 0.0f); }

    anira::InferenceHandler handler(pp, config);
    handler.prepare({static_cast<float>(k_signal_size), 44100.0f});
    handler.set_inference_backend(GetParam().backend);

    std::vector<float> audio(k_signal_size, 0.0f);
    std::array<float*, 1> ch = {audio.data()};

    // --- First inference ---
    size_t prev = handler.get_available_samples(0);
    handler.process(ch.data(), k_signal_size);
    wait_for_inference(handler, prev);

    // Model adds 1.0 to state; initial state was 0 → expect 1.0.
    for (size_t j = 0; j < k_state_size; ++j) {
        EXPECT_FLOAT_EQ(pp.get_input(1, j), 1.0f) << "state[" << j << "] wrong after 1st inference";
    }

    // --- Second inference ---
    prev = handler.get_available_samples(0);
    handler.process(ch.data(), k_signal_size);
    wait_for_inference(handler, prev);

    // State was 1.0 going in → expect 2.0 after second inference.
    for (size_t j = 0; j < k_state_size; ++j) {
        EXPECT_FLOAT_EQ(pp.get_input(1, j), 2.0f) << "state[" << j << "] wrong after 2nd inference";
    }
}

// ---------------------------------------------------------------------------
// Sine oscillator: phase state is passed between inferences
// ---------------------------------------------------------------------------

static constexpr size_t k_sine_signal_size = 512;
static constexpr float k_sample_rate = 44100.0f;

namespace {

anira::InferenceConfig make_sine_config(const Backend& b) {
    std::vector<anira::ModelData> const model_data = {
        {anira_tilde_test::model_path("sine_oscillator", b), b.backend},
    };

    std::vector<anira::TensorShape> const tensor_shapes = {
        {
            // inputs:  freq [1,1,512] (streamable, drives inference), phase [1,1]
            {{1, 1, static_cast<int64_t>(k_sine_signal_size)}, {1, 1}},
            // outputs: audio [1,1,512], phase [1,1]
            {{1, 1, static_cast<int64_t>(k_sine_signal_size)}, {1, 1}},
        },
    };

    anira::ProcessingSpec const proc_spec({1, 1},                   // preprocess_input_channels
                                          {1, 1},                   // postprocess_output_channels
                                          {k_sine_signal_size, 0},  // preprocess_input_size
                                                                    // (freq=512,
                                                                    // phase=non-streamable)
                                          {k_sine_signal_size, 0}   // postprocess_output_size
                                                                    // (audio=512,
                                                                    // phase=non-streamable)
    );

    return {model_data, tensor_shapes, proc_spec, 10.0f};
}

}  // namespace

TEST_P(StatePassing, SineOscillatorPhaseAccumulates) {
    auto config = make_sine_config(GetParam());
    // output_tensor=1 (phase out) feeds back into input_tensor=1 (phase in)
    std::vector<StatePair> const state_pairs = {{.m_output_tensor = 1, .m_input_tensor = 1}};
    StatePassingPrePostProcessor pp(config, state_pairs);

    anira::InferenceHandler handler(pp, config);
    handler.prepare({static_cast<float>(k_sine_signal_size), k_sample_rate});
    handler.set_inference_backend(GetParam().backend);

    const float freq = 440.0f;
    std::vector<float> freq_buf(k_sine_signal_size, freq);
    std::vector<float> audio_buf(k_sine_signal_size, 0.0f);
    std::array<const float*, 1> freq_ch = {freq_buf.data()};
    std::array<float*, 1> out_ch = {audio_buf.data()};

    const float phase_per_block =
        std::fmod(2.0f * std::numbers::pi_v<float> * freq / k_sample_rate *
                      static_cast<float>(k_sine_signal_size),
                  2.0f * std::numbers::pi_v<float>);

    // --- First inference ---
    size_t prev = handler.get_available_samples(0);
    handler.process(freq_ch.data(), k_sine_signal_size, out_ch.data(), k_sine_signal_size);
    wait_for_inference(handler, prev);

    const float phase_1 = pp.get_input(1, 0);
    EXPECT_NEAR(phase_1, phase_per_block, 1e-4f) << "phase wrong after 1st inference";

    // --- Second inference ---
    prev = handler.get_available_samples(0);
    handler.process(freq_ch.data(), k_sine_signal_size, out_ch.data(), k_sine_signal_size);
    wait_for_inference(handler, prev);

    const float phase_2 = pp.get_input(1, 0);
    const float expected_phase_2 =
        std::fmod(phase_1 + phase_per_block, 2.0f * std::numbers::pi_v<float>);
    EXPECT_NEAR(phase_2, expected_phase_2, 1e-4f)
        << "phase wrong after 2nd inference — state may not have been passed";
}

TEST_P(StatePassing, SineOscillatorContinuousAcrossBlockBoundaries) {
    auto config = make_sine_config(GetParam());
    std::vector<StatePair> const state_pairs = {{.m_output_tensor = 1, .m_input_tensor = 1}};
    StatePassingPrePostProcessor pp(config, state_pairs);

    anira::InferenceHandler handler(pp, config);
    handler.prepare({static_cast<float>(k_sine_signal_size), k_sample_rate});
    handler.set_inference_backend(GetParam().backend);

    const float freq = 440.0f;
    // For a continuous sine, |sample[i] - sample[i-1]| <= 2*pi*f/sr.
    // A discontinuity (state not passed) produces a jump of up to ~2.0.
    const float max_step = 2.0f * std::numbers::pi_v<float> * freq / k_sample_rate;

    std::vector<float> freq_buf(k_sine_signal_size, freq);
    std::array<const float*, 1> freq_ch = {freq_buf.data()};

    auto run_block = [&](std::vector<float>& out) {
        out.resize(k_sine_signal_size);
        std::array<float*, 1> out_ch = {out.data()};
        size_t const prev = handler.get_available_samples(0);
        handler.process(freq_ch.data(), k_sine_signal_size, out_ch.data(), k_sine_signal_size);
        wait_for_inference(handler, prev);
    };

    // Discard warm-up blocks to move past the latency-padding zeros.
    std::vector<float> discard;
    int const warmup = static_cast<int>(handler.get_latency() / k_sine_signal_size) + 1;
    for (int i = 0; i < warmup; ++i) { run_block(discard); }

    // Capture two consecutive real output blocks.
    std::vector<float> block1, block2;
    run_block(block1);
    run_block(block2);

    // Check every consecutive sample pair, including the block boundary.
    auto check = [&](float a, float b, size_t idx) {
        EXPECT_LE(std::abs(b - a), max_step * 1.1f) << "discontinuity at sample " << idx;
    };
    for (size_t i = 1; i < k_sine_signal_size; ++i) { check(block1[i - 1], block1[i], i); }
    check(block1.back(), block2.front(), k_sine_signal_size);  // ← the boundary
    for (size_t i = 1; i < k_sine_signal_size; ++i) {
        check(block2[i - 1], block2[i], k_sine_signal_size + i);
    }
}

INSTANTIATE_TEST_SUITE_P(Backends,
                         StatePassing,
                         testing::ValuesIn(anira_tilde_test::backends()),
                         anira_tilde_test::param_name);
