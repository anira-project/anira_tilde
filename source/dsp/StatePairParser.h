#pragma once

#include <iosfwd>
#include <string>
#include <vector>

struct StatePair {
    size_t output_tensor;  // global output tensor index (must be non-streamable)
    size_t input_tensor;   // global input tensor index (must be non-streamable)
};

// Parse state_pairs from a JSON stream.
// Returns an empty vector if the stream does not contain a valid state_config.
std::vector<StatePair> parse_state_pairs(std::istream& stream);

// Convenience overload that opens the file at json_path.
// Returns an empty vector if the path is empty or the file cannot be opened.
std::vector<StatePair> parse_state_pairs(const std::string& json_path);
