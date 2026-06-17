#include <gtest/gtest.h>
#include <anira_tilde/Engine.h>
#include <vector>

using namespace anira_tilde;

#ifndef SINE_OSC_JSON_PATH
#error "SINE_OSC_JSON_PATH must be defined via CMake"
#endif
#ifndef DOWNSAMPLE_JSON_PATH
#error "DOWNSAMPLE_JSON_PATH must be defined via CMake"
#endif
#ifndef MULTICHANNEL_OUT_JSON_PATH
#error "MULTICHANNEL_OUT_JSON_PATH must be defined via CMake"
#endif

#ifdef USE_LIBTORCH

namespace {

void run_one_block(Engine& engine, size_t n_in_ch, size_t n_out_ch, size_t frames) {
    std::vector<std::vector<float>> in (n_in_ch,  std::vector<float>(frames, 0.0f));
    std::vector<std::vector<float>> out(n_out_ch, std::vector<float>(frames, 0.0f));
    std::vector<const float*> in_ptrs(n_in_ch);
    std::vector<float*>       out_ptrs(n_out_ch);
    for (size_t c = 0; c < n_in_ch;  ++c) in_ptrs[c]  = in[c].data();
    for (size_t c = 0; c < n_out_ch; ++c) out_ptrs[c] = out[c].data();
    engine.process(in_ptrs.data(),  n_in_ch,
                   out_ptrs.data(), n_out_ch,
                   frames);
}

} // namespace

TEST(EngineReload, LoadingTwiceReplacesSession) {
    Engine engine;
    ASSERT_TRUE(engine.load_config(SINE_OSC_JSON_PATH));
    ASSERT_TRUE(engine.config_loaded());
    const size_t in_ch_1  = engine.layout().sig_input_channels.size();
    const size_t out_ch_1 = engine.layout().sig_output_channels.size();

    // Reload the same config. Should succeed; previous Session is destroyed.
    ASSERT_TRUE(engine.load_config(SINE_OSC_JSON_PATH));
    ASSERT_TRUE(engine.config_loaded());

    // Layout unchanged after reload of identical config.
    EXPECT_EQ(engine.layout().sig_input_channels.size(),  in_ch_1);
    EXPECT_EQ(engine.layout().sig_output_channels.size(), out_ch_1);
}

TEST(EngineReload, ReadyResetsAfterReload) {
    Engine engine;
    ASSERT_TRUE(engine.load_config(SINE_OSC_JSON_PATH));
    engine.prepare(64, 44100.0);
    EXPECT_TRUE(engine.ready());

    ASSERT_TRUE(engine.load_config(SINE_OSC_JSON_PATH));
    EXPECT_FALSE(engine.ready())
        << "load_config must park the engine until prepare() runs again";
}

TEST(EngineReload, ProcessBeforeReprepareIsSafe) {
    Engine engine;
    ASSERT_TRUE(engine.load_config(SINE_OSC_JSON_PATH));
    engine.prepare(64, 44100.0);

    ASSERT_TRUE(engine.load_config(SINE_OSC_JSON_PATH));
    // Engine is parked. process() must not crash or touch stale buffers.
    EXPECT_NO_THROW(run_one_block(engine,
                                  engine.layout().sig_input_channels.size(),
                                  engine.layout().sig_output_channels.size(),
                                  64));
}

TEST(EngineReload, ReprepareRevivesEngine) {
    Engine engine;
    ASSERT_TRUE(engine.load_config(SINE_OSC_JSON_PATH));
    engine.prepare(64, 44100.0);

    ASSERT_TRUE(engine.load_config(SINE_OSC_JSON_PATH));
    engine.prepare(64, 44100.0);
    EXPECT_TRUE(engine.ready());
    EXPECT_NO_THROW(run_one_block(engine,
                                  engine.layout().sig_input_channels.size(),
                                  engine.layout().sig_output_channels.size(),
                                  64));
}

// Regression: a single output tensor carrying multiple channels (the encode-
// style RAVE shape, e.g. 16 latents in one [1, C, 1] tensor). The dry/wet
// Mixer's per-channel delay state must be sized to the total output-channel
// count, not the output-tensor count. Sizing it to the tensor count (1) made
// every channel >= 1 index out of bounds in process_channel_block and crashed
// the audio thread with a write fault.
TEST(EngineReload, MultiChannelSingleTensorOutputDoesNotCrash) {
    Engine engine;
    ASSERT_TRUE(engine.load_config(MULTICHANNEL_OUT_JSON_PATH));

    ASSERT_EQ(engine.layout().sig_output_channels.size(), 1u)
        << "fixture must have exactly one output tensor";
    const size_t n_in_ch  = engine.layout().total_signal_inputs();
    const size_t n_out_ch = engine.layout().total_signal_outputs();
    ASSERT_GT(n_out_ch, 1u) << "fixture's single output tensor must be multi-channel";

    engine.prepare(64, 44100.0);
    ASSERT_TRUE(engine.ready());

    // Run several blocks so the latency delay line wraps and every output
    // channel's read/write indices are exercised. Pre-fix this faults on the
    // first block.
    for (int i = 0; i < 8; ++i)
        EXPECT_NO_THROW(run_one_block(engine, n_in_ch, n_out_ch, 64));
}

TEST(EngineReload, BadPathLeavesEngineEmpty) {
    Engine engine;
    std::string err;
    EXPECT_FALSE(engine.load_config("/does/not/exist.json", &err));
    EXPECT_FALSE(engine.config_loaded());
    EXPECT_FALSE(err.empty());
}

#else // !USE_LIBTORCH

TEST(EngineReload, SkippedWithoutLibTorch) {
    GTEST_SKIP() << "LibTorch unavailable — Engine reload tests skipped";
}

#endif
