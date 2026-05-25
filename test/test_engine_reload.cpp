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
    const size_t in_ch_1  = engine.sig_input_channels().size();
    const size_t out_ch_1 = engine.sig_output_channels().size();

    // Reload the same config. Should succeed; previous Session is destroyed.
    ASSERT_TRUE(engine.load_config(SINE_OSC_JSON_PATH));
    ASSERT_TRUE(engine.config_loaded());

    // Layout unchanged after reload of identical config.
    EXPECT_EQ(engine.sig_input_channels().size(),  in_ch_1);
    EXPECT_EQ(engine.sig_output_channels().size(), out_ch_1);
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
                                  engine.sig_input_channels().size(),
                                  engine.sig_output_channels().size(),
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
                                  engine.sig_input_channels().size(),
                                  engine.sig_output_channels().size(),
                                  64));
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
