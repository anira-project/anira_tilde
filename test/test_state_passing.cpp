#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <anira/anira.h>
#include "StatePassingPrePostProcessor.h"

#ifndef STATE_TEST_MODEL_PATH
#error "STATE_TEST_MODEL_PATH must be defined via CMake"
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

#else  // !USE_LIBTORCH

TEST(StatePassing, SkippedWithoutLibTorch) {
    GTEST_SKIP() << "LibTorch not available – state-passing integration test skipped";
}

#endif  // USE_LIBTORCH
