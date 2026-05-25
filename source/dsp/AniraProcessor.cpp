#include "AniraProcessor.h"

#include <filesystem>
#include <stdexcept>

static anira::ContextConfig load_context_config(anira::JsonConfigLoader& loader) {
    auto ptr = loader.get_context_config();
    if (!ptr) throw std::runtime_error("Failed to load context config (JSON may be malformed)");
    return std::move(*ptr);
}

static anira::InferenceConfig load_inference_config(anira::JsonConfigLoader& loader, const std::string& path) {
    auto ptr = loader.get_inference_config();
    if (!ptr) throw std::runtime_error("Failed to load inference config from: " + path + " (JSON may be malformed or missing 'inference_config' key)");
    auto config = std::move(*ptr);

    auto config_dir = std::filesystem::path(path).parent_path();
    for (auto& md : config.m_model_data) {
        if (!md.m_is_binary) {
            auto model_path = std::filesystem::path(std::string(static_cast<char*>(md.m_data), md.m_size));
            if (model_path.is_relative()) {
                config.set_model_path((config_dir / model_path).string(), md.m_backend);
            }
        }
    }

    // State passing requires strictly sequential inference: if two workers race,
    // pre_process(N+1) may read stale state before post_process(N) has written it.
    if (!parse_state_pairs(path).empty()) {
        config.m_num_parallel_processors = 1;
    }
    return config;
}

AniraProcessor::AniraProcessor(std::string json_config_path) :
    m_config_loader(json_config_path),
    m_anira_context(load_context_config(m_config_loader)),
    m_inference_config(load_inference_config(m_config_loader, json_config_path)),
    m_state_pairs(parse_state_pairs(json_config_path)),
    m_pp_processor(m_inference_config, m_state_pairs),
    m_inference_handler(m_pp_processor, m_inference_config)
{
    std::vector<size_t> inShapes;
    std::vector<size_t> outShapes;
    auto processing_spec = m_inference_config.m_processing_spec;

    for (int i = 0; i < m_inference_config.m_tensor_shape[0].m_tensor_input_shape.size(); ++i) {
        size_t size = 1;
        for (int j = 0; j < m_inference_config.m_tensor_shape[0].m_tensor_input_shape[i].size(); ++j) {
            size *= static_cast<size_t>(m_inference_config.m_tensor_shape[0].m_tensor_input_shape[i][j]);
        }
        inShapes.emplace_back(size);
    }

    for (int i = 0; i < m_inference_config.m_tensor_shape[0].m_tensor_output_shape.size(); ++i) {
        size_t size = 1;
        for (int j = 0; j < m_inference_config.m_tensor_shape[0].m_tensor_output_shape[i].size(); ++j) {
            size *= static_cast<size_t>(m_inference_config.m_tensor_shape[0].m_tensor_output_shape[i][j]);
        }
        outShapes.emplace_back(size);
    }

    auto inSz  = processing_spec.m_preprocess_input_size;
    auto inCh  = processing_spec.m_preprocess_input_channels;
    auto outSz = processing_spec.m_postprocess_output_size;
    auto outCh = processing_spec.m_postprocess_output_channels;

    input_sizes = inSz;
    output_sizes = outSz;

    for (int i = 0; i < (int)inShapes.size(); ++i) {
        if (inSz[i] > 0) {
            inSigCh.push_back(inCh[i]);
        } else if (is_state_input(static_cast<size_t>(i))) {
            // State input tensor: managed internally, not exposed as a Max inlet.
        } else {
            size_t channels = inCh[i];
            size_t total_elems = inShapes[i];
            std::vector<size_t> msg_ch;
            if (channels == 0) channels = 1;
            if (total_elems % channels != 0) {
                throw std::runtime_error("Inconsistent message channel configuration");
            }
            size_t elems_per_ch = total_elems / channels;
            for (size_t c = 0; c < channels; ++c) {
                msg_ch.push_back(elems_per_ch);
            }
            inMsgCh.push_back(msg_ch);
        }
    }
    for (int i = 0; i < (int)outShapes.size(); ++i) {
        if (outSz[i] > 0) {
            outSigCh.push_back(outCh[i]);
        } else if (is_state_output(static_cast<size_t>(i))) {
            // State output tensor: fed back internally, not exposed as a Max outlet.
        } else {
            size_t channels = outCh[i];
            size_t total_elems = outShapes[i];
            std::vector<size_t> msg_ch;
            if (channels == 0) channels = 1;
            if (total_elems % channels != 0) {
                throw std::runtime_error("Inconsistent message channel configuration");
            }
            size_t elems_per_ch = total_elems / channels;
            for (size_t c = 0; c < channels; ++c) {
                msg_ch.push_back(elems_per_ch);
            }
            outMsgCh.push_back(msg_ch);
        }
    }
}

void AniraProcessor::prepare(size_t buffer_size, double sample_rate)
{
    anira::HostConfig host_config {
        static_cast<float>(buffer_size),
        static_cast<float>(sample_rate),
    };

    size_t decoder_index = 0;
    unsigned int decoder_latency = 0;
    bool use_custom_latency = false;

    for (size_t i = 0; i < output_sizes.size(); ++i) {
        if (i < input_sizes.size() && input_sizes[i] < output_sizes[i]) {
            decoder_index = i;
            size_t safety_margin = (buffer_size > output_sizes[i]) ? buffer_size : output_sizes[i];
            decoder_latency = static_cast<unsigned int>(output_sizes[i] + safety_margin);
            use_custom_latency = true;
            break;
        }
    }

    if (use_custom_latency) {
        m_inference_handler.prepare(host_config, decoder_latency, decoder_index);
    } else {
        m_inference_handler.prepare(host_config);
    }

    m_selected_backend = anira::InferenceBackend::LIBTORCH;
    m_inference_handler.set_inference_backend(m_selected_backend);

    // --- Rate adaptation state ---
    size_t n_sig_in  = inSigCh.size();
    size_t n_sig_out = outSigCh.size();
    // InferenceManager iterates ALL tensor indices (signal + state), so the adj
    // arrays must cover every tensor.  State tensor entries are left at 0/nullptr:
    // their loops never execute because preprocess/postprocess_size == 0.
    size_t n_total_in  = input_sizes.size();
    size_t n_total_out = output_sizes.size();

    m_sample_pos.assign(n_sig_in, 0);

    m_hold_output.resize(n_sig_out);
    m_ds_out_buf.resize(n_sig_out);
    for (size_t i = 0; i < n_sig_out; ++i) {
        m_hold_output[i].assign(outSigCh[i], 0.0f);
        m_ds_out_buf[i].assign(outSigCh[i], 0.0f);
    }

    // Pre-allocate pointer arrays (sized to total tensor count, not just signal tensors)
    m_adj_in_ch_ptrs.resize(n_total_in);
    m_adj_in_tensor_ptrs.assign(n_total_in, nullptr);
    m_adj_in_sizes.assign(n_total_in, 0);
    for (size_t i = 0; i < n_sig_in; ++i) {
        m_adj_in_ch_ptrs[i].assign(inSigCh[i], nullptr);
        m_adj_in_tensor_ptrs[i] = m_adj_in_ch_ptrs[i].data();
    }

    m_adj_out_ch_ptrs.resize(n_total_out);
    m_adj_out_tensor_ptrs.assign(n_total_out, nullptr);
    m_adj_out_sizes.assign(n_total_out, 0);
    for (size_t i = 0; i < n_sig_out; ++i) {
        m_adj_out_ch_ptrs[i].assign(outSigCh[i], nullptr);
        m_adj_out_tensor_ptrs[i] = m_adj_out_ch_ptrs[i].data();
    }

}

size_t AniraProcessor::get_latency_samples()
{
    return m_inference_handler.get_latency();
}

size_t* AniraProcessor::process(const float* const* const* input_data, size_t* num_input_samples, float* const* const* output_data, size_t* num_output_samples)
{
    // Use signal tensor counts: state tensors are internal to the pre-post processor.
    size_t n_in  = inSigCh.size();
    size_t n_out = outSigCh.size();

    // Check whether any tensor needs rate adaptation.
    bool needs_rate_adapt = false;
    for (size_t i = 0; i < n_in && i < n_out; ++i) {
        if (is_upsample(i) || is_downsample(i)) { needs_rate_adapt = true; break; }
    }

    if (!needs_rate_adapt) {
        // State-passing: pop first so post_process(N-1) writes fresh state before
        // pre_process(N) reads it.
        if (!m_state_pairs.empty()) {
            auto* result = m_inference_handler.pop_data(output_data, num_output_samples);
            m_inference_handler.push_data(input_data, num_input_samples);
            return result;
        }
        return m_inference_handler.process(input_data, num_input_samples, output_data, num_output_samples);
    }

    // --- Build adjusted input arrays ---
    for (size_t i = 0; i < n_in; ++i) {
        size_t n      = num_input_samples[i];
        size_t num_ch = inSigCh[i];

        if (i < n_out && is_upsample(i)) {
            // Fire one inference per output_size boundary within this callback.
            size_t pos    = m_sample_pos[i];
            size_t out_sz = output_sizes[i];
            // First boundary at or after pos:
            size_t boundary = (pos % out_sz == 0) ? pos : ((pos / out_sz) + 1) * out_sz;

            if (boundary < pos + n) {
                size_t offset = boundary - pos;
                m_adj_in_sizes[i] = input_sizes[i];
                for (size_t c = 0; c < num_ch; ++c)
                    m_adj_in_ch_ptrs[i][c] = input_data[i][c] + offset;
            } else {
                m_adj_in_sizes[i] = 0;
                for (size_t c = 0; c < num_ch; ++c)
                    m_adj_in_ch_ptrs[i][c] = input_data[i][c]; // unused (size=0)
            }
            m_sample_pos[i] += n;

        } else {
            // Downsample, same-rate, or non-streamable: pass all samples through.
            m_adj_in_sizes[i] = n;
            for (size_t c = 0; c < num_ch; ++c)
                m_adj_in_ch_ptrs[i][c] = input_data[i][c];
        }
        // m_adj_in_tensor_ptrs[i] already points at m_adj_in_ch_ptrs[i].data()
    }

    // --- Build adjusted output arrays ---
    for (size_t i = 0; i < n_out; ++i) {
        size_t n      = num_output_samples[i];
        size_t num_ch = outSigCh[i];

        if (i < n_in && is_downsample(i)) {
            // Pop exactly output_sizes[i] samples into the temp buffer;
            // we'll fill the caller's buffer with the held value afterwards.
            m_adj_out_sizes[i] = output_sizes[i];
            for (size_t c = 0; c < num_ch; ++c)
                m_adj_out_ch_ptrs[i][c] = m_ds_out_buf[i].data() + c;
        } else {
            // Upsample or same-rate: pop directly into caller's buffer.
            m_adj_out_sizes[i] = n;
            for (size_t c = 0; c < num_ch; ++c)
                m_adj_out_ch_ptrs[i][c] = output_data[i][c];
        }
        // m_adj_out_tensor_ptrs[i] already points at m_adj_out_ch_ptrs[i].data()
    }

    auto* adj_input  = reinterpret_cast<const float* const* const*>(m_adj_in_tensor_ptrs.data());
    auto* adj_output = reinterpret_cast<float* const* const*>(m_adj_out_tensor_ptrs.data());

    // --- Push / pop ---
    // State-passing: pop first so post_process(N-1) writes state before push.
    if (!m_state_pairs.empty()) {
        m_inference_handler.pop_data(adj_output, m_adj_out_sizes.data());
        m_inference_handler.push_data(adj_input,  m_adj_in_sizes.data());
    } else {
        m_inference_handler.push_data(adj_input,  m_adj_in_sizes.data());
        m_inference_handler.pop_data(adj_output,  m_adj_out_sizes.data());
    }

    // --- Post-process downsample: sample-and-hold into caller's buffer ---
    for (size_t i = 0; i < n_out; ++i) {
        if (i < n_in && is_downsample(i)) {
            size_t n      = num_output_samples[i];
            size_t num_ch = outSigCh[i];
            for (size_t c = 0; c < num_ch; ++c) {
                // m_adj_out_sizes[i] is set to 0 by process_output on failure.
                if (m_adj_out_sizes[i] > 0)
                    m_hold_output[i][c] = m_ds_out_buf[i][c];
                for (size_t s = 0; s < n; ++s)
                    output_data[i][c][s] = m_hold_output[i][c];
            }
        }
    }

    return m_adj_out_sizes.data();
}

bool AniraProcessor::is_upsample(size_t i) const {
    return i < input_sizes.size() && i < output_sizes.size()
        && input_sizes[i] > 0 && output_sizes[i] > 0
        && input_sizes[i] < output_sizes[i];
}

bool AniraProcessor::is_downsample(size_t i) const {
    return i < input_sizes.size() && i < output_sizes.size()
        && input_sizes[i] > 0 && output_sizes[i] > 0
        && input_sizes[i] > output_sizes[i];
}

void AniraProcessor::set_input(const float& input, size_t i, size_t j) {
    m_pp_processor.set_input(input, i, j);
}

float AniraProcessor::get_output(size_t i, size_t j) {
    return m_pp_processor.get_output(i, j);
}

bool AniraProcessor::is_state_input(size_t tensor_index) const {
    for (const auto& pair : m_state_pairs) {
        if (pair.input_tensor == tensor_index) return true;
    }
    return false;
}

bool AniraProcessor::is_state_output(size_t tensor_index) const {
    for (const auto& pair : m_state_pairs) {
        if (pair.output_tensor == tensor_index) return true;
    }
    return false;
}
