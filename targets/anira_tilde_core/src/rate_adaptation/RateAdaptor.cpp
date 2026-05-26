#include "anira_tilde/rate_adaptation/RateAdaptor.h"

namespace anira_tilde {

static RateAdaptor::Kind classify(size_t in_sz, size_t out_sz) {
    using Kind = RateAdaptor::Kind;
    if (in_sz == 0 || out_sz == 0 || in_sz == out_sz) return Kind::Equal;
    return in_sz < out_sz ? Kind::Upsample : Kind::Downsample;
}

bool RateAdaptor::next_inference_offset(size_t pos,
                                        size_t output_block_size,
                                        size_t host_samples,
                                        size_t& out_offset) noexcept {
    // Boundary = smallest multiple of output_block_size that is ≥ pos.
    const size_t boundary = (pos % output_block_size == 0)
        ? pos
        : ((pos / output_block_size) + 1) * output_block_size;
    if (boundary >= pos + host_samples) return false;
    out_offset = boundary - pos;
    return true;
}

void RateAdaptor::prepare(const TensorLayout& layout, size_t /*host_buffer_size*/) {
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
        if (m_tensors[i].kind == Kind::Downsample) {
            m_tensors[i].downsample_hold.resize(layout.sig_output_channels[i], 1);
            m_tensors[i].downsample_pop .resize(layout.sig_output_channels[i], 1);
            m_tensors[i].downsample_hold.clear();
            m_tensors[i].downsample_pop .clear();
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
            size_t offset = 0;
            const bool fire = next_inference_offset(t.upsample_pos,
                                                    layout.output_block_sizes[i],
                                                    host_samples,
                                                    offset);
            if (fire) {
                m_in_view_sample_counts[i] = layout.input_block_sizes[i];
                for (size_t c = 0; c < num_ch; ++c)
                    m_in_view_channels[i][c] = input_data[i][c] + offset;
            } else {
                // No boundary in this block — feed nothing.
                m_in_view_sample_counts[i] = 0;
                for (size_t c = 0; c < num_ch; ++c)
                    m_in_view_channels[i][c] = input_data[i][c];
            }
            t.upsample_pos += host_samples;
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
            // Pop a single sample into our scratch; post_dispatch will
            // splatter it across the host output.
            m_out_view_sample_counts[i] = layout.output_block_sizes[i];
            for (size_t c = 0; c < num_ch; ++c)
                m_out_view_channels[i][c] = m_tensors[i].downsample_pop.get_write_pointer(c);
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
        for (size_t c = 0; c < num_ch; ++c) {
            // Refresh the held value when anira actually produced output.
            if (m_out_view_sample_counts[i] > 0)
                t.downsample_hold.set_sample(c, 0, t.downsample_pop.get_sample(c, 0));
            const float held = t.downsample_hold.get_sample(c, 0);
            for (size_t s = 0; s < host_samples; ++s)
                output_data[i][c][s] = held;
        }
    }
}

} // namespace anira_tilde
