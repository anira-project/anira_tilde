#pragma once

    #include <c74_min.h>
#include <memory>
#include <vector>
#include <string>

#include <anira_tilde/anira_tilde.h>

class AniraTilde : public c74::min::object<AniraTilde>, public c74::min::vector_operator<> {
public:
    explicit AniraTilde(const c74::min::atoms& args = {});
    ~AniraTilde() override = default;

    MIN_DESCRIPTION {
        "Neural network inference wrapper for Max. "
        "The anira~ external integrates the <a href='https://github.com/anira-project/anira'>anira</a> library to offer neural network inference inside Max. "
        "Inlets and outlets are dynamically configured during object initialization streamable (signal) and non-streamable (message) data. "
        "The last outlet always outputs the processing latency."
    };
    MIN_TAGS    { "audio, ML, inference" };
    MIN_AUTHOR  { "Valentin Ackva, Fares Schulz, Konstantin Fontaine" };
    MIN_RELATED { "nn~" };

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

    void operator()(c74::min::audio_bundle input, c74::min::audio_bundle output) override;

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

    // Max audio is double-precision, anira_tilde_core is float-native.
    // Cast scratch + pointer arrays sized in dspsetup so the audio thread
    // never allocates.
    anira::Buffer<float>       m_in_float_scratch;   // channels × buffer_size
    anira::Buffer<float>       m_out_float_scratch;
    std::vector<const float*>  m_in_float_ptrs;
    std::vector<float*>        m_out_float_ptrs;
};
