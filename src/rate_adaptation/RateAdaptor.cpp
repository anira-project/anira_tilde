#include "anira_tilde/rate_adaptation/RateAdaptor.h"

namespace anira_tilde {

void RateAdaptor::prepare(const TensorLayout& layout, size_t /*host_buffer_size*/) {
    const size_t n_sig_in    = layout.sig_input_channels.size();
    const size_t n_sig_out   = layout.sig_output_channels.size();
    const size_t n_total_in  = layout.input_block_sizes.size();
    const size_t n_total_out = layout.output_block_sizes.size();

    m_kinds.assign(std::max(n_sig_in, n_sig_out), Kind::Equal);
    m_active = false;
    for (size_t i = 0; i < n_sig_in && i < n_sig_out; ++i) {
        const size_t in_sz  = layout.input_block_sizes[i];
        const size_t out_sz = layout.output_block_sizes[i];
        if (in_sz == 0 || out_sz == 0 || in_sz == out_sz)
            m_kinds[i] = Kind::Equal;
        else if (in_sz < out_sz)
            m_kinds[i] = Kind::Upsample;
        else
            m_kinds[i] = Kind::Downsample;
        if (m_kinds[i] != Kind::Equal) m_active = true;
    }

    m_upsample_input_position.assign(n_sig_in, 0);

    m_downsample_held_samples.clear();
    m_downsample_pop_buffer.clear();
    m_downsample_held_samples.reserve(n_sig_out);
    m_downsample_pop_buffer.reserve(n_sig_out);
    for (size_t i = 0; i < n_sig_out; ++i) {
        m_downsample_held_samples.emplace_back(layout.sig_output_channels[i], 1);
        m_downsample_pop_buffer.emplace_back(layout.sig_output_channels[i], 1);
    }

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

void RateAdaptor::pre_dispatch(const TensorLayout& layout,
                               const float* const* const* input_data,  const size_t* num_input_samples,
                               float* const* const*       output_data, const size_t* num_output_samples) {
    const size_t n_in  = layout.sig_input_channels.size();
    const size_t n_out = layout.sig_output_channels.size();

    // Input views.
    for (size_t i = 0; i < n_in; ++i) {
        const size_t host_samples = num_input_samples[i];
        const size_t num_ch       = layout.sig_input_channels[i];

        if (m_kinds[i] == Kind::Upsample) {
            // Fire one inference per output_size boundary within this host block.
            const size_t pos      = m_upsample_input_position[i];
            const size_t out_sz   = layout.output_block_sizes[i];
            const size_t boundary = (pos % out_sz == 0) ? pos
                                                        : ((pos / out_sz) + 1) * out_sz;

            if (boundary < pos + host_samples) {
                const size_t offset = boundary - pos;
                m_in_view_sample_counts[i] = layout.input_block_sizes[i];
                for (size_t c = 0; c < num_ch; ++c)
                    m_in_view_channels[i][c] = input_data[i][c] + offset;
            } else {
                m_in_view_sample_counts[i] = 0;
                for (size_t c = 0; c < num_ch; ++c)
                    m_in_view_channels[i][c] = input_data[i][c];
            }
            m_upsample_input_position[i] += host_samples;
        } else {
            m_in_view_sample_counts[i] = host_samples;
            for (size_t c = 0; c < num_ch; ++c)
                m_in_view_channels[i][c] = input_data[i][c];
        }
    }

    // Output views.
    for (size_t i = 0; i < n_out; ++i) {
        const size_t host_samples = num_output_samples[i];
        const size_t num_ch       = layout.sig_output_channels[i];

        if (i < n_in && m_kinds[i] == Kind::Downsample) {
            // Pop a single sample into our scratch; post_dispatch() will
            // splatter it across the host output.
            m_out_view_sample_counts[i] = layout.output_block_sizes[i];
            for (size_t c = 0; c < num_ch; ++c)
                m_out_view_channels[i][c] = m_downsample_pop_buffer[i].get_write_pointer(c);
        } else {
            m_out_view_sample_counts[i] = host_samples;
            for (size_t c = 0; c < num_ch; ++c)
                m_out_view_channels[i][c] = output_data[i][c];
        }
    }
}

void RateAdaptor::post_dispatch(const TensorLayout& layout,
                                float* const* const* output_data,
                                const size_t* num_output_samples) {
    const size_t n_in  = layout.sig_input_channels.size();
    const size_t n_out = layout.sig_output_channels.size();
    for (size_t i = 0; i < n_out; ++i) {
        if (i >= n_in || m_kinds[i] != Kind::Downsample) continue;

        const size_t host_samples = num_output_samples[i];
        const size_t num_ch       = layout.sig_output_channels[i];
        for (size_t c = 0; c < num_ch; ++c) {
            if (m_out_view_sample_counts[i] > 0)
                m_downsample_held_samples[i].set_sample(c, 0,
                    m_downsample_pop_buffer[i].get_sample(c, 0));

            const float held = m_downsample_held_samples[i].get_sample(c, 0);
            for (size_t s = 0; s < host_samples; ++s)
                output_data[i][c][s] = held;
        }
    }
}

} // namespace anira_tilde
