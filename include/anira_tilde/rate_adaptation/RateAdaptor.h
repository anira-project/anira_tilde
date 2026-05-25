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
 *
 * The "view" vectors below present the host's pointers to anira in the shape
 * anira's push_data/pop_data want (per-tensor channel-pointer arrays). Their
 * contents are rewritten per process() call but their storage is allocated
 * once in prepare() so the audio thread never allocates.
 */
class ANIRA_TILDE_API RateAdaptor {
public:
    void prepare(const TensorLayout& layout, size_t host_buffer_size);

    bool is_active() const noexcept { return m_active; }

    /// Build the views anira sees this block. Call before dispatching to
    /// anira; pre_dispatch() reads the host pointers and rewrites the
    /// internal view arrays accessible via input_views() / output_views().
    void pre_dispatch(const TensorLayout& layout,
                      const float* const* const* input_data,  const size_t* num_input_samples,
                      float* const* const*       output_data, const size_t* num_output_samples);

    /// Apply post-dispatch fixups (downsample sample-and-hold).
    /// Call after anira has filled the output views.
    void post_dispatch(const TensorLayout& layout,
                       float* const* const* output_data,
                       const size_t* num_output_samples);

    // Views to feed anira's push_data / pop_data with.
    const float* const* const* input_views()              const { return reinterpret_cast<const float* const* const*>(m_in_view_tensors.data()); }
    float* const* const*       output_views()             const { return reinterpret_cast<float* const* const*>      (m_out_view_tensors.data()); }
    size_t*                    input_view_sample_counts()  { return m_in_view_sample_counts.data(); }
    size_t*                    output_view_sample_counts() { return m_out_view_sample_counts.data(); }

private:
    enum class Kind : uint8_t { Equal, Upsample, Downsample };

    // Precomputed in prepare().
    std::vector<Kind>                 m_kinds;                     // per signal tensor
    bool                              m_active = false;

    // Upsample state.
    std::vector<size_t>               m_upsample_input_position;   // running counter per tensor

    // Downsample state.
    std::vector<anira::Buffer<float>> m_downsample_held_samples;   // [tensor] (channels × 1)
    std::vector<anira::Buffer<float>> m_downsample_pop_buffer;     // [tensor] (channels × 1)

    // What we present to anira this block (rewritten per process() call).
    std::vector<std::vector<const float*>> m_in_view_channels;     // [tensor][channel]
    std::vector<const float**>             m_in_view_tensors;      // [tensor] = channels.data()
    std::vector<size_t>                    m_in_view_sample_counts;// [tensor]
    std::vector<std::vector<float*>>       m_out_view_channels;
    std::vector<float**>                   m_out_view_tensors;
    std::vector<size_t>                    m_out_view_sample_counts;
};

} // namespace anira_tilde
