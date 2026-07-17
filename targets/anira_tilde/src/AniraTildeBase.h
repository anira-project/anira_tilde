#pragma once

#include <c74_min.h>
#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <anira_tilde/anira_tilde.h>

// Shared implementation for the anira~ and mc.anira~ Max objects. The two objects
// are identical except for how their signal ports are laid out (one mono port per
// channel vs. one multichannel port per tensor) and the Max audio base they
// derive from. Everything else — message handling, config loading, the engine,
// the double<->float scratch, message/latency ports — lives here.
//
// CRTP: Derived is the concrete object class; AudioBase is the min audio operator
// it should derive from (vector_operator<> for anira~, mc_operator<> for
// mc.anira~). Derived supplies create_signal_ports() and, after construction,
// calls initialize() to load its config and build its ports.
template <class Derived, class AudioBase>
class AniraTildeBase : public c74::min::object<Derived>, public AudioBase {
public:
    enum MaxType {
        SIGNAL  = 0,
        MESSAGE = 1,
        LATENCY = 2,
    };

    struct Input {
        std::unique_ptr<c74::min::inlet<>> inlet;
        MaxType type;
        size_t  tensor_index;
        size_t  num_channels;
    };

    struct Output {
        std::unique_ptr<c74::min::outlet<>> outlet;
        MaxType type;
        size_t  tensor_index;
        size_t  num_channels;
    };

    void operator()(c74::min::audio_bundle input, c74::min::audio_bundle output) {
        if (!m_engine.config_loaded()) return;

        const size_t n_in   = m_total_in_channels;
        const size_t n_out  = m_total_out_channels;
        const size_t frames = static_cast<size_t>(input.frame_count());

        const size_t in_avail  = static_cast<size_t>(input.channel_count());
        const size_t out_avail = static_cast<size_t>(output.channel_count());

        // Cast Max's double input into float scratch. The model expects exactly
        // n_in channels; ignore extra incoming channels and zero-pad missing
        // ones. (For the per-channel anira~ all n_in are always present, so this
        // is a plain copy; mc.anira~ relies on the pad/clamp on a mismatch.)
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

protected:
    // name is the Max object name ("anira~" / "mc.anira~"), used in console output.
    explicit AniraTildeBase(const char* name) :
        m_name(name),
        dry_wet(this, "mix", "Set the dry/wet mix of the output",
            MIN_FUNCTION {
                if (!m_engine.config_loaded() || m_engine.mixing_disabled()) {
                    c74::max::error("%s: Mix parameter disabled for this model configuration.", m_name);
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

                    // Allocate the float scratch we cast Max's double audio into /
                    // out of each block. Sized by total channels across all signal
                    // tensors (one entry per channel, regardless of port layout).
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
                    c74::max::error("%s: External not initialized. Activate DSP.", m_name);
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
    {}

    // Load the config given as the first object argument and, if valid, build the
    // object's inlets/outlets. Called from the Derived constructor body (not the
    // base ctor) so the create_signal_ports() override dispatches correctly.
    void initialize(const c74::min::atoms& args) {
        const std::string json_path = get_json_path(args);
        if (!m_valid_config_submitted) return;

        std::string err;
        if (!m_engine.load_config(json_path, &err)) {
            c74::max::object_error(nullptr, "%s: Failed to load config: %s", m_name, err.c_str());
            return;
        }

        c74::max::post(
            "%s: Signal input tensors: %zu, Signal output tensors: %zu, "
            "Message input tensors: %zu, Message output tensors: %zu",
            m_name,
            m_engine.layout().sig_input_channels.size(),
            m_engine.layout().sig_output_channels.size(),
            m_engine.layout().msg_input_channels.size(),
            m_engine.layout().msg_output_channels.size()
        );
        if (m_engine.layout().state_pairs.size() > 0) {
            c74::max::post("%s: State-passing mode active — %zu state pair(s) fed back internally.",
                           m_name, m_engine.layout().state_pairs.size());
        }

        init_external(m_engine.layout().sig_input_channels,
                      m_engine.layout().sig_output_channels,
                      m_engine.layout().msg_input_channels,
                      m_engine.layout().msg_output_channels);
    }

    // Build the signal inlets/outlets. The only part that differs between the two
    // objects, so it is supplied by Derived. num_channels on each created port is
    // used for the multichannel-output negotiation in mc.anira~.
    virtual void create_signal_ports(const std::vector<size_t>& sig_inputs,
                                     const std::vector<size_t>& sig_outputs) = 0;

    // Helper for Derived's create_signal_ports: a descriptive inlet/outlet label.
    static std::string channel_label(const char* kind, size_t tensor, size_t channel) {
        return std::string("(") + kind + ") Tensor " + std::to_string(tensor + 1)
             + ", Channel " + std::to_string(channel + 1);
    }
    static std::string tensor_label(const char* kind, size_t tensor) {
        return std::string("(") + kind + ") Tensor " + std::to_string(tensor + 1);
    }

    const char* m_name;

    std::vector<Input>  m_sig_inlets;
    std::vector<Input>  m_msg_inlets;
    std::vector<Output> m_sig_outlets;
    std::vector<Output> m_msg_outlets;

    // These must be public: min-api detects the audio "dspsetup" hook via
    // `decltype(&Derived::dspsetup)` and calls `m_min_object.dspsetup(args)`
    // from outside the class, both of which require public access. A protected
    // dspsetup makes min-api's has_dspsetup SFINAE silently fail, so the custom
    // DSP setup never runs and the audio scratch is never allocated.
public:
    c74::min::message<> dry_wet;
    c74::min::message<> dspsetup;
    c74::min::message<> anything;
    c74::min::message<> m_float;
    c74::min::message<> m_int;
    c74::min::message<> bang;

protected:
    anira_tilde::Engine m_engine;
    bool        m_valid_config_submitted = false;
    bool        m_initialized            = false;
    std::string m_config_file_path;

    // Total channel counts summed across all signal tensors; the scratch and the
    // engine's flat channel arrays are sized from these.
    size_t m_total_in_channels  = 0;
    size_t m_total_out_channels = 0;

    // Max audio is double-precision, anira_tilde_core is float-native. Cast
    // scratch + pointer arrays sized in dspsetup so the audio thread never allocates.
    anira::Buffer<float>       m_in_float_scratch;   // total_channels × buffer_size
    anira::Buffer<float>       m_out_float_scratch;
    std::vector<const float*>  m_in_float_ptrs;
    std::vector<float*>        m_out_float_ptrs;

private:
    std::string get_json_path(const c74::min::atoms& args) {
        if (args.size() > 0) {
            const std::string input_path = static_cast<std::string>(args[0]);
            c74::min::path p(input_path);
            if (p) {
                m_config_file_path = static_cast<std::string>(p);
                c74::max::post("%s: Loading config from: %s", m_name, m_config_file_path.c_str());
                m_valid_config_submitted = true;
                return m_config_file_path;
            }
            return "";
        }
        c74::max::post("%s: No config file specified", m_name);
        return "";
    }

    void parse_input_messages(int inlet_num, const std::vector<float>& args) {
        if (!m_engine.config_loaded()) return;
        const size_t num_sig_inputs = m_sig_inlets.size();
        const size_t msg_index = static_cast<size_t>(inlet_num) - num_sig_inputs;
        const size_t expected = m_msg_inlets[msg_index].num_channels;
        if (args.size() != expected) {
            c74::max::error("%s: Incorrect number of elements for inlet %zu. Expected %zu but got %zu.",
                            m_name, msg_index + 1, expected, args.size());
            return;
        }
        m_engine.set_message_input(m_msg_inlets[msg_index].tensor_index, args);
    }

    void send_latency_outlet(size_t latency_samples) {
        for (const auto& outlet : m_msg_outlets) {
            if (outlet.type == MaxType::LATENCY) {
                outlet.outlet->send(static_cast<int>(latency_samples));
                break;
            }
        }
    }

    // Build all inlets/outlets: signal ports (delegated to Derived), then the
    // shared message ports and the trailing latency outlet.
    void init_external(const std::vector<size_t>& sig_inputs,
                       const std::vector<size_t>& sig_outputs,
                       const std::vector<std::vector<size_t>>& msg_inputs,
                       const std::vector<std::vector<size_t>>& msg_outputs) {
        m_sig_inlets.clear();
        m_msg_inlets.clear();
        m_sig_outlets.clear();
        m_msg_outlets.clear();

        m_total_in_channels  = std::accumulate(sig_inputs.begin(),  sig_inputs.end(),  size_t{0});
        m_total_out_channels = std::accumulate(sig_outputs.begin(), sig_outputs.end(), size_t{0});

        create_signal_ports(sig_inputs, sig_outputs);

        // Message inlets: tensor_index continues past the signal tensors.
        const size_t msg_in_base = sig_inputs.size();
        for (size_t t = 0; t < msg_inputs.size(); ++t) {
            for (size_t c = 0; c < msg_inputs[t].size(); ++c) {
                Input in;
                in.inlet        = std::make_unique<c74::min::inlet<>>(this, channel_label("message", t, c), "");
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
                out.outlet       = std::make_unique<c74::min::outlet<>>(this, channel_label("message", t, c), "");
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
};
