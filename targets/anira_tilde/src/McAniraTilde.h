#pragma once

#include <c74_min.h>
#include <memory>
#include <vector>
#include <string>

#include <anira_tilde/anira_tilde.h>

// Multichannel sibling of anira~. Identical inference behaviour, but each signal
// tensor is presented as a single Max multichannel (mc) signal instead of one
// mono inlet/outlet per channel. Inheriting mc_operator<> sets the Z_MC_INLETS
// flag so the signal inlets accept packed multichannel signals; the mc outputs
// are declared via the multichanneloutputs/inputchanged methods added in
// maxclass_setup (min-api does not provide these).
class McAniraTilde : public c74::min::object<McAniraTilde>,
                     public c74::min::mc_operator<> {
public:
    explicit McAniraTilde(const c74::min::atoms& args = {});
    ~McAniraTilde() override = default;

    MIN_DESCRIPTION {
        "Multichannel neural network inference wrapper for Max. "
        "Like <o>anira~</o>, but each signal tensor flows on a single multichannel "
        "(mc) signal cord rather than one mono inlet/outlet per channel. "
        "Inlets and outlets are dynamically configured during object initialization "
        "for streamable (signal) and non-streamable (message) data. "
        "The last outlet always outputs the processing latency."
    };
    MIN_TAGS    { "audio, ML, inference, multichannel" };
    MIN_AUTHOR  { "Valentin Ackva, Fares Schulz, Konstantin Fontaine" };
    MIN_RELATED { "anira~, nn~" };

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

    std::vector<Input>  m_sig_inlets;
    std::vector<Input>  m_msg_inlets;
    std::vector<Output> m_sig_outlets;
    std::vector<Output> m_msg_outlets;

    c74::min::message<> dry_wet;
    c74::min::message<> dspsetup;
    c74::min::message<> anything;
    c74::min::message<> m_float;
    c74::min::message<> m_int;
    c74::min::message<> bang;
    c74::min::message<> maxclass_setup;

    // Called by the generic Min perform trampoline (mc_operator has no pure-virtual
    // perform of its own). With Z_MC_INLETS set, the audio_bundles carry the packed
    // multichannel signals.
    void operator()(c74::min::audio_bundle input, c74::min::audio_bundle output);

    // A_CANT methods Max calls to negotiate multichannel signal connections.
    // multichanneloutputs reports how many channels a given signal outlet emits;
    // inputchanged is notified when an inlet's incoming channel count changes.
    static long mc_multichanneloutputs(c74::max::t_object* x, long index);
    static long mc_inputchanged(c74::max::t_object* x, long index, long count);

private:
    std::string get_json_path(const c74::min::atoms& args);
    void parse_input_messages(int inlet_num, const std::vector<float>& args);
    void init_external(const std::vector<size_t>& sig_inputs,
                       const std::vector<size_t>& sig_outputs,
                       const std::vector<std::vector<size_t>>& msg_inputs,
                       const std::vector<std::vector<size_t>>& msg_outputs);
    void send_latency_outlet(size_t latency_samples);

    anira_tilde::Engine m_engine;
    bool        m_valid_config_submitted = false;
    bool        m_initialized            = false;
    std::string m_config_file_path;

    // Total channel counts summed across all signal tensors. Unlike anira~,
    // m_sig_inlets/m_sig_outlets now hold one entry per tensor, so the float
    // scratch and the engine's flat channel arrays are sized from these totals.
    size_t m_total_in_channels  = 0;
    size_t m_total_out_channels = 0;

    // Max audio is double-precision, anira_tilde_core is float-native.
    // Cast scratch + pointer arrays sized in dspsetup so the audio thread
    // never allocates.
    anira::Buffer<float>       m_in_float_scratch;   // total_channels × buffer_size
    anira::Buffer<float>       m_out_float_scratch;
    std::vector<const float*>  m_in_float_ptrs;
    std::vector<float*>        m_out_float_ptrs;
};
