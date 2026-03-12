#include <gtest/gtest.h>
#include <sstream>
#include "StatePairParser.h"

// ---- parse_state_pairs(istream) ----

TEST(ParseStatePairs, EmptyStream) {
    std::istringstream ss("");
    auto pairs = parse_state_pairs(ss);
    EXPECT_TRUE(pairs.empty());
}

TEST(ParseStatePairs, MalformedJson) {
    std::istringstream ss("{not valid json");
    auto pairs = parse_state_pairs(ss);
    EXPECT_TRUE(pairs.empty());
}

TEST(ParseStatePairs, NoStateConfigKey) {
    std::istringstream ss(R"({"inference_config": {}})");
    auto pairs = parse_state_pairs(ss);
    EXPECT_TRUE(pairs.empty());
}

TEST(ParseStatePairs, StateConfigWithoutStatePairs) {
    std::istringstream ss(R"({"state_config": {}})");
    auto pairs = parse_state_pairs(ss);
    EXPECT_TRUE(pairs.empty());
}

TEST(ParseStatePairs, SinglePair) {
    std::istringstream ss(R"({
        "state_config": {
            "state_pairs": [
                {"output_tensor": 1, "input_tensor": 2}
            ]
        }
    })");
    auto pairs = parse_state_pairs(ss);
    ASSERT_EQ(pairs.size(), 1u);
    EXPECT_EQ(pairs[0].output_tensor, 1u);
    EXPECT_EQ(pairs[0].input_tensor, 2u);
}

TEST(ParseStatePairs, MultiplePairs) {
    std::istringstream ss(R"({
        "state_config": {
            "state_pairs": [
                {"output_tensor": 1, "input_tensor": 2},
                {"output_tensor": 3, "input_tensor": 4}
            ]
        }
    })");
    auto pairs = parse_state_pairs(ss);
    ASSERT_EQ(pairs.size(), 2u);
    EXPECT_EQ(pairs[0].output_tensor, 1u);
    EXPECT_EQ(pairs[0].input_tensor, 2u);
    EXPECT_EQ(pairs[1].output_tensor, 3u);
    EXPECT_EQ(pairs[1].input_tensor, 4u);
}

TEST(ParseStatePairs, PairMissingOutputTensor) {
    std::istringstream ss(R"({
        "state_config": {
            "state_pairs": [
                {"input_tensor": 2}
            ]
        }
    })");
    auto pairs = parse_state_pairs(ss);
    EXPECT_TRUE(pairs.empty());
}

TEST(ParseStatePairs, PairMissingInputTensor) {
    std::istringstream ss(R"({
        "state_config": {
            "state_pairs": [
                {"output_tensor": 1}
            ]
        }
    })");
    auto pairs = parse_state_pairs(ss);
    EXPECT_TRUE(pairs.empty());
}

TEST(ParseStatePairs, SkipsMalformedPairAmongValid) {
    std::istringstream ss(R"({
        "state_config": {
            "state_pairs": [
                {"output_tensor": 1, "input_tensor": 2},
                {"output_tensor": 3},
                {"output_tensor": 5, "input_tensor": 6}
            ]
        }
    })");
    auto pairs = parse_state_pairs(ss);
    ASSERT_EQ(pairs.size(), 2u);
    EXPECT_EQ(pairs[0].output_tensor, 1u);
    EXPECT_EQ(pairs[1].output_tensor, 5u);
}

// ---- parse_state_pairs(path) ----

TEST(ParseStatePairsPath, EmptyPath) {
    auto pairs = parse_state_pairs(std::string{});
    EXPECT_TRUE(pairs.empty());
}

TEST(ParseStatePairsPath, NonExistentFile) {
    auto pairs = parse_state_pairs("/tmp/anira_tilde_test_nonexistent_file.json");
    EXPECT_TRUE(pairs.empty());
}
