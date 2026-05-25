#pragma once

#include <vector>
#include <cstddef>

namespace anira_tilde {

/// Compute the effective buffer size to pass to Session::prepare.
///
/// For upsampling tensors (input_size < output_size) the host must drive the
/// processor with a buffer equal to the model's input_size, not host_buffer_size,
/// to avoid buffer overflow.  Non-streaming tensors (input_size == 0) are skipped.
inline size_t compute_effective_buffer_size(
    const std::vector<size_t>& input_sizes,
    const std::vector<size_t>& output_sizes,
    size_t host_buffer_size)
{
    if (input_sizes.empty()) return host_buffer_size;

    size_t effective = 0;
    for (size_t i = 0; i < input_sizes.size(); ++i) {
        if (i >= output_sizes.size()) continue;   // no paired output — skip
        if (input_sizes[i] == 0) continue;        // non-streaming tensor — skip

        size_t current = host_buffer_size;
        if (input_sizes[i] < output_sizes[i]) {
            current = input_sizes[i];
        }
        if (current > effective) effective = current;
    }
    if (effective == 0) effective = host_buffer_size;
    return effective;
}

} // namespace anira_tilde
