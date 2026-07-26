#include "anira_tilde/rate_adaptation/RateAdaptor.h"

#include <algorithm>

namespace anira_tilde {

static RateAdaptor::Kind classify(size_t in_sz, size_t out_sz) {
    using Kind = RateAdaptor::Kind;
    if (in_sz == 0 || out_sz == 0 || in_sz == out_sz) return Kind::Equal;
    return in_sz < out_sz ? Kind::Upsample : Kind::Downsample;
}

void RateAdaptor::prepare(const TensorLayout& layout, size_t max_block_size) {
    const size_t n_sig_in    = layout.sig_input_channels.size();
    const size_t n_sig_out   = layout.sig_output_channels.size();
    const size_t n_total_in  = layout.input_block_sizes.size();
    const size_t n_total_out = layout.output_block_sizes.size();
    const size_t n_pairs     = std::max(n_sig_in, n_sig_out);

    // Per-tensor state in one place. Only the slot matching `kind` is
    // actually populated; others stay default-constructed.
    m_tensors.clear();
    m_tensors.resize(n_pairs);
    m_active = false;
    for (size_t i = 0; i < n_sig_in && i < n_sig_out; ++i) {
        m_tensors[i].kind = classify(layout.input_block_sizes[i],
                                     layout.output_block_sizes[i]);
        if (m_tensors[i].kind == Kind::Upsample) {
            // Worst case one inference per output-block boundary in a
            // max-sized host block, +1 for the block-straddling boundary.
            const size_t max_fires =
                max_block_size / layout.output_block_sizes[i] + 1;
            m_tensors[i].max_fires = max_fires;
            m_tensors[i].upsample_gather.assign(
                layout.sig_input_channels[i],
                std::vector<float>(max_fires * layout.input_block_sizes[i], 0.0f));
        }
        if (m_tensors[i].kind == Kind::Downsample) {
            const size_t max_fires =
                max_block_size / layout.input_block_sizes[i] + 1;
            m_tensors[i].max_fires = max_fires;
            m_tensors[i].downsample_hold.resize(layout.sig_output_channels[i], 1);
            m_tensors[i].downsample_pop .resize(layout.sig_output_channels[i],
                                                max_fires * layout.output_block_sizes[i]);
            m_tensors[i].downsample_hold.clear();
            m_tensors[i].downsample_pop .clear();
            m_tensors[i].downsample_offsets.reserve(max_fires);
        }
        if (m_tensors[i].kind != Kind::Equal) m_active = true;
    }

    // View storage sized to the *total* tensor count so anira's per-tensor
    // loops iterate cleanly; state/message slots stay null with size=0
    // (anira skips them).
    m_in_view_channels.resize(n_total_in);
    m_in_view_tensors.assign(n_total_in, nullptr);
    m_in_view_sample_counts.assign(n_total_in, 0);
    for (size_t i = 0; i < n_sig_in; ++i) {
        m_in_view_channels[i].assign(layout.sig_input_channels[i], nullptr);
        m_in_view_tensors[i] = m_in_view_channels[i].data();
    }

    m_out_view_channels.resize(n_total_out);
    m_out_view_tensors.assign(n_total_out, nullptr);
    m_out_view_sample_counts.assign(n_total_out, 0);
    for (size_t i = 0; i < n_sig_out; ++i) {
        m_out_view_channels[i].assign(layout.sig_output_channels[i], nullptr);
        m_out_view_tensors[i] = m_out_view_channels[i].data();
    }
}

RateAdaptor::AniraView RateAdaptor::pre_dispatch(const TensorLayout& layout,
                                                 const float* const* const* input_data,
                                                 const size_t* num_input_samples,
                                                 float* const* const* output_data,
                                                 const size_t* num_output_samples) {
    const size_t n_in  = layout.sig_input_channels.size();
    const size_t n_out = layout.sig_output_channels.size();

    // Input views.
    for (size_t i = 0; i < n_in; ++i) {
        PerTensor&   t            = m_tensors[i];
        const size_t host_samples = num_input_samples[i];
        const size_t num_ch       = layout.sig_input_channels[i];

        if (t.kind == Kind::Upsample) {
            // Gather one input block per output-size boundary crossed inside
            // this host block; anira runs one inference per gathered block.
            const size_t out_block = layout.output_block_sizes[i];
            const size_t in_block  = layout.input_block_sizes[i];
            const size_t start     = t.pos;
            size_t boundary = (start % out_block == 0)
                ? start
                : ((start / out_block) + 1) * out_block;
            size_t fires = 0;
            for (; boundary < start + host_samples &&
                   fires < t.max_fires; boundary += out_block) {
                const size_t offset = boundary - start;
                for (size_t c = 0; c < num_ch; ++c) {
                    float* dst = t.upsample_gather[c].data() + fires * in_block;
                    for (size_t j = 0; j < in_block; ++j)
                        dst[j] = input_data[i][c][std::min(offset + j, host_samples - 1)];
                }
                ++fires;
            }
            m_in_view_sample_counts[i] = fires * in_block;
            for (size_t c = 0; c < num_ch; ++c)
                m_in_view_channels[i][c] = t.upsample_gather[c].data();
            t.pos += host_samples;
        } else {
            // Equal-rate or downsample: hand the whole host buffer to anira.
            m_in_view_sample_counts[i] = host_samples;
            for (size_t c = 0; c < num_ch; ++c)
                m_in_view_channels[i][c] = input_data[i][c];
        }
    }

    // Output views.
    for (size_t i = 0; i < n_out; ++i) {
        const size_t host_samples = num_output_samples[i];
        const size_t num_ch       = layout.sig_output_channels[i];

        if (i < n_in && m_tensors[i].kind == Kind::Downsample) {
            // One inference completes per input_size samples of host time:
            // pop one output block per boundary crossed in this host block
            // into our scratch; post_dispatch holds each across its segment.
            PerTensor&   t        = m_tensors[i];
            const size_t in_block = layout.input_block_sizes[i];
            const size_t start    = t.pos;
            size_t boundary = (start % in_block == 0)
                ? start
                : ((start / in_block) + 1) * in_block;
            t.downsample_offsets.clear();
            for (; boundary < start + host_samples &&
                   t.downsample_offsets.size() < t.max_fires; boundary += in_block)
                t.downsample_offsets.push_back(boundary - start);
            t.pos += host_samples;

            m_out_view_sample_counts[i] =
                t.downsample_offsets.size() * layout.output_block_sizes[i];
            for (size_t c = 0; c < num_ch; ++c)
                m_out_view_channels[i][c] = t.downsample_pop.get_write_pointer(c);
        } else {
            m_out_view_sample_counts[i] = host_samples;
            for (size_t c = 0; c < num_ch; ++c)
                m_out_view_channels[i][c] = output_data[i][c];
        }
    }

    return {
        reinterpret_cast<const float* const* const*>(m_in_view_tensors.data()),
        m_in_view_sample_counts.data(),
        reinterpret_cast<float* const* const*>(m_out_view_tensors.data()),
        m_out_view_sample_counts.data(),
    };
}

void RateAdaptor::post_dispatch(const TensorLayout& layout,
                                float* const* const* output_data,
                                const size_t* num_output_samples) {
    const size_t n_in  = layout.sig_input_channels.size();
    const size_t n_out = layout.sig_output_channels.size();
    for (size_t i = 0; i < n_out; ++i) {
        if (i >= n_in || m_tensors[i].kind != Kind::Downsample) continue;

        PerTensor&   t            = m_tensors[i];
        const size_t host_samples = num_output_samples[i];
        const size_t num_ch       = layout.sig_output_channels[i];
        const size_t out_block    = layout.output_block_sizes[i];
        // anira's pop_data rewrites the requested count in place: 0 means the
        // receive ring couldn't cover the request and our scratch holds
        // zero-fill, not results — keep the previous held value then.
        const size_t fires        = m_out_view_sample_counts[i] == 0
            ? 0
            : t.downsample_offsets.size();
        for (size_t c = 0; c < num_ch; ++c) {
            size_t fire = 0;
            float  held = t.downsample_hold.get_sample(c, 0);
            for (size_t s = 0; s < host_samples; ++s) {
                // Advance to the pop whose boundary this sample has reached;
                // each pop's first sample becomes the new held value.
                while (fire < fires && s >= t.downsample_offsets[fire]) {
                    held = t.downsample_pop.get_sample(c, fire * out_block);
                    ++fire;
                }
                output_data[i][c][s] = held;
            }
            t.downsample_hold.set_sample(c, 0, held);
        }
    }
}

} // namespace anira_tilde
