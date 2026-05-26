#pragma once

#include <anira/anira.h>
#include <cstdint>
#include <vector>

#include "anira_tilde/Exports.h"
#include "anira_tilde/inference/TensorLayout.h"
#include "anira_tilde/state_passing/StatePairParser.h"

namespace anira_tilde {

/**
 * @brief Per-tensor sample-rate adaptation on top of anira's inference handler.
 *
 * The audio host always presents the engine with a single host-rate buffer per
 * tensor. anira's inference, however, may have been configured with a different
 * block size per tensor (model expects e.g. 1 latent in / 2048 audio out).
 * This class adapts between the two.
 *
 * For each signal tensor we precompute a Kind:
 *   - Equal      — tensor runs at host rate. Pass the host buffer through
 *                  unchanged.
 *   - Upsample   — model input block is smaller than the host buffer
 *                  (input_size < output_size). Within each host block we
 *                  fire exactly one inference when we cross the next
 *                  output_size boundary; anira fills the larger output
 *                  buffer across subsequent callbacks.
 *   - Downsample — model input block is larger than the host buffer
 *                  (input_size > output_size). We hand anira every host
 *                  input sample (it accumulates), and ask it to pop one
 *                  output sample into a 1-sample scratch buffer; we then
 *                  sample-and-hold that value across the host output.
 */
class ANIRA_TILDE_API RateAdaptor {
public:
    enum class Kind : uint8_t { Equal, Upsample, Downsample };

    /// Bundle of pointer arrays in the shape anira's push_data / pop_data
    /// want. Returned by pre_dispatch() and consumed by the caller's
    /// InferenceHandler invocation.
    struct AniraView {
        const float* const* const* in_tensors;
        size_t*                    in_sample_counts;
        float* const* const*       out_tensors;
        size_t*                    out_sample_counts;
    };

    void prepare(const TensorLayout& layout, size_t host_buffer_size);

    bool is_active() const noexcept { return m_active; }

    /// Compute the views anira sees this block.
    AniraView pre_dispatch(const TensorLayout& layout,
                           const float* const* const* input_data,  const size_t* num_input_samples,
                           float* const* const*       output_data, const size_t* num_output_samples);

    /// Apply post-dispatch fixups (downsample sample-and-hold).
    void post_dispatch(const TensorLayout& layout,
                       float* const* const* output_data,
                       const size_t* num_output_samples);

private:
    /// State for a single signal tensor pair. Only the slot matching `kind`
    /// is actually used; the others stay default-constructed (zero cost).
    struct PerTensor {
        Kind                 kind         = Kind::Equal;
        size_t               upsample_pos = 0;     // running input counter (Upsample)
        anira::Buffer<float> downsample_hold;      // last seen output (Downsample)
        anira::Buffer<float> downsample_pop;       // 1-sample scratch anira writes into (Downsample)
    };

    /// Returns the offset into this host block where the next inference
    /// boundary falls, or std::nullopt if no boundary is crossed.
    /// `pos` is the running input-sample counter for this tensor.
    static bool next_inference_offset(size_t pos,
                                      size_t output_block_size,
                                      size_t host_samples,
                                      size_t& out_offset) noexcept;

    std::vector<PerTensor> m_tensors;    // one per max(n_sig_in, n_sig_out)
    bool                   m_active = false;

    // anira-shaped view storage. Rewritten per pre_dispatch() call,
    // allocated once in prepare(). Channels-of-tensor[t] live in
    // m_in_view_channels[t]; m_in_view_tensors[t] points at that
    // sub-vector's data().
    std::vector<std::vector<const float*>> m_in_view_channels;
    std::vector<const float**>             m_in_view_tensors;
    std::vector<size_t>                    m_in_view_sample_counts;
    std::vector<std::vector<float*>>       m_out_view_channels;
    std::vector<float**>                   m_out_view_tensors;
    std::vector<size_t>                    m_out_view_sample_counts;
};

} // namespace anira_tilde
