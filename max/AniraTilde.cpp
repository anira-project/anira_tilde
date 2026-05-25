#include "AniraTilde.h"

#include <algorithm>

AniraTilde::AniraTilde(const c74::min::atoms& args) :
    dry_wet(this, "mix", "Set the dry/wet mix of the output",
        MIN_FUNCTION {
            if (!m_engine.config_loaded() || m_engine.mixing_disabled()) {
                c74::max::error("anira~: Mix parameter disabled for this model configuration.");
                return {};
            }
            const float new_mix = std::clamp(static_cast<float>(args[0]), 0.0f, 100.0f) / 100.0f;
            m_engine.set_dry_wet_mix(new_mix);
            return {};
        }
    ),
    dspsetup(this, "dspsetup",
        MIN_FUNCTION {
            const auto sample_rate = static_cast<double>(args[0]);
            const auto buffer_size = static_cast<size_t>(args[1]);
            if (m_engine.config_loaded()) {
                m_engine.prepare(buffer_size, sample_rate);
                send_latency_outlet(m_engine.latency_samples());

                // Allocate the float scratch we'll cast Max's double audio
                // into / out of each block.
                m_in_float_scratch.resize (m_sig_inlets.size(),  buffer_size);
                m_out_float_scratch.resize(m_sig_outlets.size(), buffer_size);
                m_in_float_ptrs.resize (m_sig_inlets.size());
                m_out_float_ptrs.resize(m_sig_outlets.size());
                for (size_t c = 0; c < m_sig_inlets.size();  ++c) m_in_float_ptrs[c]  = m_in_float_scratch.get_read_pointer(c);
                for (size_t c = 0; c < m_sig_outlets.size(); ++c) m_out_float_ptrs[c] = m_out_float_scratch.get_write_pointer(c);
            }
            m_initialized = true;
            return {};
        }
    ),
    bang(this, "bang", "Output non-streamable parameters",
        MIN_FUNCTION {
            if (!m_initialized || !m_engine.config_loaded()) {
                c74::max::error("anira~: External not initialized. Activate DSP.");
                return {};
            }

            for (const auto& outlet : m_msg_outlets) {
                if (outlet.type != MaxType::MESSAGE) continue;
                std::vector<float> data;
                data.reserve(outlet.num_channels);
                for (size_t j = 0; j < outlet.num_channels; ++j)
                    data.push_back(m_engine.get_message_output(outlet.tensor_index, j));
                if (data.size() == 1) {
                    outlet.outlet->send(data[0]);
                } else {
                    c74::min::atoms msg;
                    msg.push_back("list");
                    for (float v : data) msg.push_back(v);
                    outlet.outlet->send(msg);
                }
            }

            for (const auto& outlet : m_msg_outlets) {
                if (outlet.type == MaxType::LATENCY) {
                    outlet.outlet->send(static_cast<int>(m_engine.latency_samples()));
                }
            }
            return {};
        }
    ),
    anything(this, "list", "Receive lists in message inlets",
        MIN_FUNCTION {
            std::vector<float> data;
            data.reserve(args.size());
            for (const auto& a : args) data.push_back(static_cast<float>(a));
            parse_input_messages(inlet, data);
            return {};
        }
    ),
    m_float(this, "float", "Receive float in message inlets",
        MIN_FUNCTION {
            parse_input_messages(inlet, { static_cast<float>(args[0]) });
            return {};
        }
    ),
    m_int(this, "int", "Receive int in message inlets",
        MIN_FUNCTION {
            parse_input_messages(inlet, { static_cast<float>(args[0]) });
            return {};
        }
    )
{
    const std::string json_path = get_json_path(args);
    if (!m_valid_config_submitted) return;

    std::string err;
    if (!m_engine.load_config(json_path, &err)) {
        c74::max::object_error(nullptr, "anira~: Failed to load config: %s", err.c_str());
        return;
    }

    c74::max::post(
        "anira~: Signal input tensors: %zu, Signal output tensors: %zu, "
        "Message input tensors: %zu, Message output tensors: %zu",
        m_engine.sig_input_channels().size(),
        m_engine.sig_output_channels().size(),
        m_engine.msg_input_channels().size(),
        m_engine.msg_output_channels().size()
    );
    if (m_engine.state_pair_count() > 0) {
        c74::max::post("anira~: State-passing mode active — %zu state pair(s) fed back internally.",
                       m_engine.state_pair_count());
    }

    init_external(m_engine.sig_input_channels(),
                  m_engine.sig_output_channels(),
                  m_engine.msg_input_channels(),
                  m_engine.msg_output_channels());
}

void AniraTilde::operator()(c74::min::audio_bundle input, c74::min::audio_bundle output) {
    if (!m_engine.config_loaded()) return;

    const size_t n_in  = m_in_float_ptrs.size();
    const size_t n_out = m_out_float_ptrs.size();
    const size_t frames = static_cast<size_t>(input.frame_count());

    // Cast Max's double input into float scratch.
    for (size_t c = 0; c < n_in; ++c) {
        const double* src = input.samples(c);
        float* dst        = m_in_float_scratch.get_write_pointer(c);
        for (size_t s = 0; s < frames; ++s) dst[s] = static_cast<float>(src[s]);
    }

    m_engine.process(m_in_float_ptrs.data(),  n_in,
                      m_out_float_ptrs.data(), n_out,
                      frames);

    // Cast float result back into Max's double output.
    for (size_t c = 0; c < n_out; ++c) {
        const float* src = m_out_float_scratch.get_read_pointer(c);
        double* dst      = output.samples(c);
        for (size_t s = 0; s < frames; ++s) dst[s] = static_cast<double>(src[s]);
    }
}

std::string AniraTilde::get_json_path(const c74::min::atoms& args) {
    if (args.size() > 0) {
        const std::string input_path = static_cast<std::string>(args[0]);
        c74::min::path p(input_path);
        if (p) {
            m_config_file_path = static_cast<std::string>(p);
            c74::max::post("anira~: Loading config from: %s", m_config_file_path.c_str());
            m_valid_config_submitted = true;
            return m_config_file_path;
        }
        return "";
    }
    c74::max::post("anira~: No config file specified");
    return "";
}

void AniraTilde::parse_input_messages(int inlet_num, const std::vector<float>& args) {
    if (!m_engine.config_loaded()) return;
    const size_t num_sig_inputs = m_sig_inlets.size();
    const size_t msg_index = static_cast<size_t>(inlet_num) - num_sig_inputs;
    const size_t expected = m_msg_inlets[msg_index].num_channels;
    if (args.size() != expected) {
        c74::max::error("anira~: Incorrect number of elements for inlet %zu. Expected %zu but got %zu.",
                        msg_index + 1, expected, args.size());
        return;
    }
    m_engine.set_message_input(m_msg_inlets[msg_index].tensor_index, args);
}

void AniraTilde::send_latency_outlet(size_t latency_samples) {
    for (const auto& outlet : m_msg_outlets) {
        if (outlet.type == MaxType::LATENCY) {
            outlet.outlet->send(static_cast<int>(latency_samples));
            break;
        }
    }
}

void AniraTilde::init_external(const std::vector<size_t>& sig_inputs,
                               const std::vector<size_t>& sig_outputs,
                               const std::vector<std::vector<size_t>>& msg_inputs,
                               const std::vector<std::vector<size_t>>& msg_outputs) {
    m_sig_inlets.clear();
    m_msg_inlets.clear();
    m_sig_outlets.clear();
    m_msg_outlets.clear();

    size_t last_audio_input_tensor  = 0;
    size_t last_audio_output_tensor = 0;

    for (size_t i = 0; i < sig_inputs.size(); ++i) {
        last_audio_input_tensor++;
        for (size_t j = 0; j < sig_inputs[i]; ++j) {
            Input in;
            in.inlet = std::make_unique<c74::min::inlet<>>(
                this,
                "(signal) Tensor " + std::to_string(i + 1) + ", Channel " + std::to_string(j + 1),
                "signal");
            in.type         = MaxType::SIGNAL;
            in.num_channels = 1;
            in.tensor_index = i;
            m_sig_inlets.push_back(std::move(in));
        }
    }

    for (size_t i = 0; i < sig_outputs.size(); ++i) {
        last_audio_output_tensor++;
        for (size_t j = 0; j < sig_outputs[i]; ++j) {
            Output out;
            out.outlet = std::make_unique<c74::min::outlet<>>(
                this,
                "(signal) Tensor " + std::to_string(i + 1) + ", Channel " + std::to_string(j + 1),
                "signal");
            out.type         = MaxType::SIGNAL;
            out.num_channels = 1;
            out.tensor_index = i;
            m_sig_outlets.push_back(std::move(out));
        }
    }

    for (size_t tensor = 0; tensor < msg_inputs.size(); ++tensor) {
        for (size_t channel = 0; channel < msg_inputs[tensor].size(); ++channel) {
            Input in;
            in.inlet = std::make_unique<c74::min::inlet<>>(
                this,
                "(message) Tensor " + std::to_string(tensor + 1) + ", Channel " + std::to_string(channel + 1),
                "");
            in.type         = MaxType::MESSAGE;
            in.num_channels = msg_inputs[tensor][channel];
            in.tensor_index = tensor + last_audio_input_tensor;
            m_msg_inlets.push_back(std::move(in));
        }
    }

    for (size_t tensor = 0; tensor < msg_outputs.size(); ++tensor) {
        for (size_t channel = 0; channel < msg_outputs[tensor].size(); ++channel) {
            Output out;
            out.outlet = std::make_unique<c74::min::outlet<>>(
                this,
                "(message) Tensor " + std::to_string(tensor + 1) + ", Channel " + std::to_string(channel + 1),
                "");
            out.type         = MaxType::MESSAGE;
            out.num_channels = msg_outputs[tensor][channel];
            out.tensor_index = tensor + last_audio_output_tensor;
            m_msg_outlets.push_back(std::move(out));
        }
    }

    Output latency_out;
    latency_out.outlet = std::make_unique<c74::min::outlet<>>(this, "(int) Latency Output", "int");
    latency_out.type         = MaxType::LATENCY;
    latency_out.num_channels = 1;
    m_msg_outlets.push_back(std::move(latency_out));
}

MIN_EXTERNAL_CUSTOM(AniraTilde, anira~);
