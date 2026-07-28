#include <anira_tilde/Engine.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "TestBackends.h"

using namespace anira_tilde;

using anira_tilde_test::Backend;

// Runs once per compiled-in backend (see TestBackends.h); LibTorch coverage
// exists only when the build enables ANIRA_WITH_LIBTORCH.
class EngineReload : public testing::TestWithParam<Backend> {};

namespace {

void run_one_block(Engine& engine, size_t n_in_ch, size_t n_out_ch, size_t frames) {
    std::vector<std::vector<float>> in(n_in_ch, std::vector<float>(frames, 0.0f));
    std::vector<std::vector<float>> out(n_out_ch, std::vector<float>(frames, 0.0f));
    std::vector<const float*> in_ptrs(n_in_ch);
    std::vector<float*> out_ptrs(n_out_ch);
    for (size_t c = 0; c < n_in_ch; ++c) { in_ptrs[c] = in[c].data(); }
    for (size_t c = 0; c < n_out_ch; ++c) { out_ptrs[c] = out[c].data(); }
    engine.process(in_ptrs.data(), n_in_ch, out_ptrs.data(), n_out_ch, frames);
}

}  // namespace

TEST_P(EngineReload, LoadingTwiceReplacesSession) {
    Engine engine;
    ASSERT_TRUE(
        engine.load_config(anira_tilde_test::json_path("sine_oscillator_test", GetParam())));
    ASSERT_TRUE(engine.config_loaded());
    const size_t in_ch_1 = engine.layout().m_sig_input_channels.size();
    const size_t out_ch_1 = engine.layout().m_sig_output_channels.size();

    // Reload the same config. Should succeed; previous Session is destroyed.
    ASSERT_TRUE(
        engine.load_config(anira_tilde_test::json_path("sine_oscillator_test", GetParam())));
    ASSERT_TRUE(engine.config_loaded());

    // Layout unchanged after reload of identical config.
    EXPECT_EQ(engine.layout().m_sig_input_channels.size(), in_ch_1);
    EXPECT_EQ(engine.layout().m_sig_output_channels.size(), out_ch_1);
}

TEST_P(EngineReload, ReadyResetsAfterReload) {
    Engine engine;
    ASSERT_TRUE(
        engine.load_config(anira_tilde_test::json_path("sine_oscillator_test", GetParam())));
    engine.prepare(64, 44100.0);
    EXPECT_TRUE(engine.ready());

    ASSERT_TRUE(
        engine.load_config(anira_tilde_test::json_path("sine_oscillator_test", GetParam())));
    EXPECT_FALSE(engine.ready()) << "load_config must park the engine until prepare() runs again";
}

TEST_P(EngineReload, ProcessBeforeReprepareIsSafe) {
    Engine engine;
    ASSERT_TRUE(
        engine.load_config(anira_tilde_test::json_path("sine_oscillator_test", GetParam())));
    engine.prepare(64, 44100.0);

    ASSERT_TRUE(
        engine.load_config(anira_tilde_test::json_path("sine_oscillator_test", GetParam())));
    // Engine is parked. process() must not crash or touch stale buffers.
    EXPECT_NO_THROW(run_one_block(engine,
                                  engine.layout().m_sig_input_channels.size(),
                                  engine.layout().m_sig_output_channels.size(),
                                  64));
}

TEST_P(EngineReload, ReprepareRevivesEngine) {
    Engine engine;
    ASSERT_TRUE(
        engine.load_config(anira_tilde_test::json_path("sine_oscillator_test", GetParam())));
    engine.prepare(64, 44100.0);

    ASSERT_TRUE(
        engine.load_config(anira_tilde_test::json_path("sine_oscillator_test", GetParam())));
    engine.prepare(64, 44100.0);
    EXPECT_TRUE(engine.ready());
    EXPECT_NO_THROW(run_one_block(engine,
                                  engine.layout().m_sig_input_channels.size(),
                                  engine.layout().m_sig_output_channels.size(),
                                  64));
}

// Regression: a single output tensor carrying multiple channels (the encode-
// style RAVE shape, e.g. 16 latents in one [1, C, 1] tensor). The Engine's
// per-channel output path must be sized to the total output-channel count,
// not the output-tensor count — sizing to the tensor count (1) once made
// every channel >= 1 index out of bounds and crashed the audio thread with
// a write fault (originally in the since-removed dry/wet Mixer).
TEST_P(EngineReload, MultiChannelSingleTensorOutputDoesNotCrash) {
    Engine engine;
    ASSERT_TRUE(
        engine.load_config(anira_tilde_test::json_path("multichannel_out_test", GetParam())));

    ASSERT_EQ(engine.layout().m_sig_output_channels.size(), 1u)
        << "fixture must have exactly one output tensor";
    const size_t n_in_ch = engine.layout().total_signal_inputs();
    const size_t n_out_ch = engine.layout().total_signal_outputs();
    ASSERT_GT(n_out_ch, 1u) << "fixture's single output tensor must be multi-channel";

    engine.prepare(64, 44100.0);
    ASSERT_TRUE(engine.ready());

    // Run several blocks so the latency delay line wraps and every output
    // channel's read/write indices are exercised. Pre-fix this faults on the
    // first block.
    for (int i = 0; i < 8; ++i) { EXPECT_NO_THROW(run_one_block(engine, n_in_ch, n_out_ch, 64)); }
}

TEST(EngineReloadErrors, BadPathLeavesEngineEmpty) {
    Engine engine;
    std::string err;
    EXPECT_FALSE(engine.load_config("/does/not/exist.json", &err));
    EXPECT_FALSE(engine.config_loaded());
    EXPECT_FALSE(err.empty());
}

INSTANTIATE_TEST_SUITE_P(Backends,
                         EngineReload,
                         testing::ValuesIn(anira_tilde_test::backends()),
                         anira_tilde_test::param_name);
