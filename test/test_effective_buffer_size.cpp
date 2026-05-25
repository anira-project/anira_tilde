#include <gtest/gtest.h>
#include "anira_tilde/rate_adaptation/EffectiveBufferSize.h"

using namespace anira_tilde;

// No upsampling: effective size equals host buffer size.
TEST(EffectiveBufferSize, NoUpsampling) {
    EXPECT_EQ(compute_effective_buffer_size({512}, {512}, 64), 64u);
}

// Upsampling only: constrained to the model input size.
TEST(EffectiveBufferSize, UpsampleOnly) {
    EXPECT_EQ(compute_effective_buffer_size({1}, {16}, 64), 1u);
}

// Downsampling: input > output, no constraint applied.
TEST(EffectiveBufferSize, DownsampleOnly) {
    EXPECT_EQ(compute_effective_buffer_size({32}, {1}, 16), 16u);
}

// No inputs at all: fall back to host buffer size.
TEST(EffectiveBufferSize, EmptyInputs) {
    EXPECT_EQ(compute_effective_buffer_size({}, {}, 64), 64u);
}

// THE BUG CASE: upsampling signal tensor + non-streaming state tensor.
// input_sizes=[1,0], output_sizes=[16,0], host_buffer=64.
// The state tensor (size 0) must not inflate effective_buffer_size back to 64.
TEST(EffectiveBufferSize, UpsamplePlusNonStreamingStateTensor) {
    EXPECT_EQ(compute_effective_buffer_size({1, 0}, {16, 0}, 64), 1u);
}

// More inputs than outputs (extra non-streaming inputs): must not inflate.
TEST(EffectiveBufferSize, MoreInputsThanOutputs) {
    EXPECT_EQ(compute_effective_buffer_size({1, 4}, {16}, 64), 1u);
}
