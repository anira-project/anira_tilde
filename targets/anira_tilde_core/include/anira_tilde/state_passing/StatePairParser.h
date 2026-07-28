#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include "anira_tilde/Exports.h"

namespace anira_tilde {

struct StatePair {
    size_t m_output_tensor;  // global output tensor index (must be non-streamable)
    size_t m_input_tensor;   // global input tensor index (must be non-streamable)
};

// Parse state_pairs from a JSON stream.
// Returns an empty vector if the stream does not contain a valid state_config.
ANIRA_TILDE_API std::vector<StatePair> parse_state_pairs(std::istream& stream);

// Convenience overload that opens the file at json_path.
// Returns an empty vector if the path is empty or the file cannot be opened.
ANIRA_TILDE_API std::vector<StatePair> parse_state_pairs(const std::string& json_path);

}  // namespace anira_tilde
