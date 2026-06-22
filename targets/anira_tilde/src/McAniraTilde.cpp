#include "McAniraTilde.h"

#include <algorithm>
#include <numeric>

McAniraTilde::McAniraTilde(const c74::min::atoms& args) :
    dry_wet(this, "mix", "Set the dry/wet mix of the output",
        MIN_FUNCTION {
            if (!m_engine.config_loaded() || m_engine.mixing_disabled()) {
                c74::max::error("mc.anira~: Mix parameter disabled for this model configuration.");
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
                // into / out of each block. Sized by total channels across all
                // signal tensors, since each inlet/outlet now carries a whole
                // tensor's worth of multichannel signal.
                m_in_float_scratch.resize (m_total_in_channels,  buffer_size);
                m_out_float_scratch.resize(m_total_out_channels, buffer_size);
                m_in_float_ptrs.resize (m_total_in_channels);
                m_out_float_ptrs.resize(m_total_out_channels);
                for (size_t c = 0; c < m_total_in_channels;  ++c) m_in_float_ptrs[c]  = m_in_float_scratch.get_read_pointer(c);
                for (size_t c = 0; c < m_total_out_channels; ++c) m_out_float_ptrs[c] = m_out_float_scratch.get_write_pointer(c);
            }
            m_initialized = true;
            return {};
        }
    ),
    bang(this, "bang", "Output non-streamable parameters",
        MIN_FUNCTION {
            if (!m_initialized || !m_engine.config_loaded()) {
                c74::max::error("mc.anira~: External not initialized. Activate DSP.");
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
    ),
    maxclass_setup(this, "maxclass_setup",
        MIN_FUNCTION {
            // min-api has no multichannel-output support, so register the two
            // A_CANT methods Max needs to negotiate mc outlets ourselves. The
            // class pointer is handed to us as the first argument.
            c74::max::t_class* c = args[0];
            c74::max::class_addmethod(c, reinterpret_cast<c74::max::method>(mc_multichanneloutputs),
                                      "multichanneloutputs", c74::max::A_CANT, 0);
            c74::max::class_addmethod(c, reinterpret_cast<c74::max::method>(mc_inputchanged),
                                      "inputchanged", c74::max::A_CANT, 0);
            return {};
        }
    )
{
    const std::string json_path = get_json_path(args);
    if (!m_valid_config_submitted) return;

    std::string err;
    if (!m_engine.load_config(json_path, &err)) {
        c74::max::object_error(nullptr, "mc.anira~: Failed to load config: %s", err.c_str());
        return;
    }

    c74::max::post(
        "mc.anira~: Signal input tensors: %zu, Signal output tensors: %zu, "
        "Message input tensors: %zu, Message output tensors: %zu",
        m_engine.layout().sig_input_channels.size(),
        m_engine.layout().sig_output_channels.size(),
        m_engine.layout().msg_input_channels.size(),
        m_engine.layout().msg_output_channels.size()
    );
    if (m_engine.layout().state_pairs.size() > 0) {
        c74::max::post("mc.anira~: State-passing mode active — %zu state pair(s) fed back internally.",
                       m_engine.layout().state_pairs.size());
    }

    init_external(m_engine.layout().sig_input_channels,
                  m_engine.layout().sig_output_channels,
                  m_engine.layout().msg_input_channels,
                  m_engine.layout().msg_output_channels);
}

void McAniraTilde::operator()(c74::min::audio_bundle input, c74::min::audio_bundle output) {
    if (!m_engine.config_loaded()) return;

    const size_t n_in  = m_total_in_channels;
    const size_t n_out = m_total_out_channels;
    const size_t frames = static_cast<size_t>(input.frame_count());

    const size_t in_avail  = static_cast<size_t>(input.channel_count());
    const size_t out_avail = static_cast<size_t>(output.channel_count());

    // Cast Max's double input into float scratch. The model expects exactly
    // n_in channels; ignore any extra incoming channels and zero-pad missing
    // ones (a channel-count mismatch is reported once from inputchanged).
    for (size_t c = 0; c < n_in; ++c) {
        float* dst = m_in_float_scratch.get_write_pointer(c);
        if (c < in_avail) {
            const double* src = input.samples(c);
            for (size_t s = 0; s < frames; ++s) dst[s] = static_cast<float>(src[s]);
        } else {
            std::fill_n(dst, frames, 0.0f);
        }
    }

    m_engine.process(m_in_float_ptrs.data(),  n_in,
                      m_out_float_ptrs.data(), n_out,
                      frames);

    // Cast float result back into Max's double output.
    const size_t out_n = std::min(n_out, out_avail);
    for (size_t c = 0; c < out_n; ++c) {
        const float* src = m_out_float_scratch.get_read_pointer(c);
        double* dst      = output.samples(c);
        for (size_t s = 0; s < frames; ++s) dst[s] = static_cast<double>(src[s]);
    }
}

std::string McAniraTilde::get_json_path(const c74::min::atoms& args) {
    if (args.size() > 0) {
        const std::string input_path = static_cast<std::string>(args[0]);
        c74::min::path p(input_path);
        if (p) {
            m_config_file_path = static_cast<std::string>(p);
            c74::max::post("mc.anira~: Loading config from: %s", m_config_file_path.c_str());
            m_valid_config_submitted = true;
            return m_config_file_path;
        }
        return "";
    }
    c74::max::post("mc.anira~: No config file specified");
    return "";
}

void McAniraTilde::parse_input_messages(int inlet_num, const std::vector<float>& args) {
    if (!m_engine.config_loaded()) return;
    const size_t num_sig_inputs = m_sig_inlets.size();
    const size_t msg_index = static_cast<size_t>(inlet_num) - num_sig_inputs;
    const size_t expected = m_msg_inlets[msg_index].num_channels;
    if (args.size() != expected) {
        c74::max::error("mc.anira~: Incorrect number of elements for inlet %zu. Expected %zu but got %zu.",
                        msg_index + 1, expected, args.size());
        return;
    }
    m_engine.set_message_input(m_msg_inlets[msg_index].tensor_index, args);
}

void McAniraTilde::send_latency_outlet(size_t latency_samples) {
    for (const auto& outlet : m_msg_outlets) {
        if (outlet.type == MaxType::LATENCY) {
            outlet.outlet->send(static_cast<int>(latency_samples));
            break;
        }
    }
}

void McAniraTilde::init_external(const std::vector<size_t>& sig_inputs,
                                 const std::vector<size_t>& sig_outputs,
                                 const std::vector<std::vector<size_t>>& msg_inputs,
                                 const std::vector<std::vector<size_t>>& msg_outputs) {
    m_sig_inlets.clear();
    m_msg_inlets.clear();
    m_sig_outlets.clear();
    m_msg_outlets.clear();

    m_total_in_channels  = std::accumulate(sig_inputs.begin(),  sig_inputs.end(),  size_t{0});
    m_total_out_channels = std::accumulate(sig_outputs.begin(), sig_outputs.end(), size_t{0});

    const auto label = [](const char* kind, size_t tensor) {
        return std::string("(mc ") + kind + ") Tensor " + std::to_string(tensor + 1);
    };
    const auto msg_label = [](const char* kind, size_t tensor, size_t channel) {
        return std::string("(") + kind + ") Tensor " + std::to_string(tensor + 1)
             + ", Channel " + std::to_string(channel + 1);
    };

    // Signal inlets: one multichannel inlet per signal input tensor, carrying
    // all of that tensor's channels on a single mc signal.
    for (size_t t = 0; t < sig_inputs.size(); ++t) {
        Input in;
        in.inlet        = std::make_unique<c74::min::inlet<>>(this, label("signal", t), "signal");
        in.type         = MaxType::SIGNAL;
        in.num_channels = sig_inputs[t];
        in.tensor_index = t;
        m_sig_inlets.push_back(std::move(in));
    }

    // Signal outlets: one multichannel outlet per signal output tensor.
    for (size_t t = 0; t < sig_outputs.size(); ++t) {
        Output out;
        out.outlet       = std::make_unique<c74::min::outlet<>>(this, label("signal", t), "signal");
        out.type         = MaxType::SIGNAL;
        out.num_channels = sig_outputs[t];
        out.tensor_index = t;
        m_sig_outlets.push_back(std::move(out));
    }

    // Message inlets: tensor_index continues past the signal tensors.
    const size_t msg_in_base = sig_inputs.size();
    for (size_t t = 0; t < msg_inputs.size(); ++t) {
        for (size_t c = 0; c < msg_inputs[t].size(); ++c) {
            Input in;
            in.inlet        = std::make_unique<c74::min::inlet<>>(this, msg_label("message", t, c), "");
            in.type         = MaxType::MESSAGE;
            in.num_channels = msg_inputs[t][c];
            in.tensor_index = msg_in_base + t;
            m_msg_inlets.push_back(std::move(in));
        }
    }

    // Message outlets: same scheme.
    const size_t msg_out_base = sig_outputs.size();
    for (size_t t = 0; t < msg_outputs.size(); ++t) {
        for (size_t c = 0; c < msg_outputs[t].size(); ++c) {
            Output out;
            out.outlet       = std::make_unique<c74::min::outlet<>>(this, msg_label("message", t, c), "");
            out.type         = MaxType::MESSAGE;
            out.num_channels = msg_outputs[t][c];
            out.tensor_index = msg_out_base + t;
            m_msg_outlets.push_back(std::move(out));
        }
    }

    // Dedicated latency outlet always at the end of the message-outlet vector.
    Output latency_out;
    latency_out.outlet       = std::make_unique<c74::min::outlet<>>(this, "(int) Latency Output", "int");
    latency_out.type         = MaxType::LATENCY;
    latency_out.num_channels = 1;
    m_msg_outlets.push_back(std::move(latency_out));
}

// Report the channel count for a given signal outlet. Signal outlets are created
// first, so the index Max passes maps directly onto m_sig_outlets.
long McAniraTilde::mc_multichanneloutputs(c74::max::t_object* x, long index) {
    auto& self = reinterpret_cast<c74::min::minwrap<McAniraTilde>*>(x)->m_min_object;
    if (index >= 0 && static_cast<size_t>(index) < self.m_sig_outlets.size())
        return static_cast<long>(self.m_sig_outlets[index].num_channels);
    return 0;
}

// Notified when the channel count arriving on a signal inlet changes. Our output
// channel counts are fixed by the model config and never depend on the input, so
// we return false (no DSP-graph re-derivation needed). We only use this to warn
// the user when the connection doesn't match what the model expects.
long McAniraTilde::mc_inputchanged(c74::max::t_object* x, long index, long count) {
    auto& self = reinterpret_cast<c74::min::minwrap<McAniraTilde>*>(x)->m_min_object;
    if (index >= 0 && static_cast<size_t>(index) < self.m_sig_inlets.size()) {
        const size_t expected = self.m_sig_inlets[index].num_channels;
        if (count > 0 && static_cast<size_t>(count) != expected) {
            c74::max::object_warn(x,
                "mc.anira~: signal inlet %ld has %ld channel(s) but the model expects %zu; "
                "extra channels are ignored and missing channels are zero-padded.",
                index + 1, count, expected);
        }
    }
    return 0;
}

// Registration. mc.anira~ must live in its own translation unit (and its own
// binary) because min-api keeps the registered class in a per-TU `this_class`
// global and only supports one class per TU — anira~ has the symmetric entry in
// AniraTilde.cpp.
namespace {
    void register_mc_anira(void* r) {
        c74::min::wrap_as_max_external<McAniraTilde>("McAniraTilde", "mc.anira~", r);
    }
}

#ifdef __APPLE__
// macOS: built as mc_anira_tilde_impl.dylib, dlopen'd by the mc.anira~ loader
// shim, which calls this C-linkage entry once it has cleared the libtorch guard.
extern "C" __attribute__((visibility("default"))) void mc_anira_tilde_impl_main(void* r) {
    register_mc_anira(r);
}
#else
// Other platforms load this module (mc.anira~) directly.
void ext_main(void* r) {
    register_mc_anira(r);
}
#endif
