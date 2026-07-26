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
 *   - Upsample   — model input block is smaller than the model output block
 *                  (input_size < output_size). One inference is due per
 *                  output_size samples of host time: within each host block
 *                  we gather one input block at every output_size boundary
 *                  crossed (0, 1, or many, depending on how the host block
 *                  compares to the output block) and feed them all to anira.
 *                  Each of the input_size samples in a block represents
 *                  output_size/input_size samples of host time (e.g. 8
 *                  latent frames per 1024-sample block = one frame per 128
 *                  samples), so the gather picks samples at that stride —
 *                  not consecutively — from the boundary onward.
 *   - Downsample — model input block is larger than the model output block
 *                  (input_size > output_size). We hand anira every host
 *                  input sample (it accumulates; one inference completes per
 *                  input_size samples) and pop one output block per
 *                  input_size boundary crossed in this host block. Each of
 *                  the output_size samples in a popped block represents
 *                  input_size/output_size samples of host time, so it is
 *                  sample-and-held across its own sub-segment (a phase
 *                  counter carries partially-played blocks across host
 *                  callbacks).
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

    /// `max_block_size` is the largest per-call sample count process() will
    /// see (model-domain when the session resamples); it bounds the upsample
    /// gather scratch.
    void prepare(const TensorLayout& layout, size_t max_block_size);

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
        Kind                 kind       = Kind::Equal;
        size_t               pos        = 0;       // running host-sample counter
        size_t               max_fires  = 0;       // scratch capacity, in inferences
        size_t               hold_phase = 0;       // host samples since the held block's boundary
        std::vector<std::vector<float>> upsample_gather;  // per-channel gathered input blocks
        anira::Buffer<float> downsample_hold;      // currently-playing output block (Downsample)
        anira::Buffer<float> downsample_pop;       // pop scratch anira writes into (Downsample)
        std::vector<size_t>  downsample_offsets;   // boundary offsets of this block's pops
    };

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
