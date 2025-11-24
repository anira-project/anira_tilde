#ifndef ANIRALIBRARY_TILDE_H
#define ANIRALIBRARY_TILDE_H

#include <c74_min.h>
#include <map>
#include <vector>
#include <string>
#include "utils/Mixer.h"

#include "dsp/AniraProcessor.h"

class AniraTilde : public c74::min::object<AniraTilde>, public c74::min::vector_operator<> {
public:
    explicit AniraTilde(const c74::min::atoms& args = {});
    ~AniraTilde() override;

    MIN_DESCRIPTION { 
        "Neural network inference wrapper for Max. "
        "The anira~ external integrates the <a href='https://github.com/anira-project/anira'>anira</a> library to offer neural network inference inside Max. "
        "It currently supports the following inference engines: LibTorch, ONNXRuntime, and TensorFlow Lite. "
        "Configuration files are read at object initialization to dynamically set inlets and outlets for the external." 
    };
    MIN_TAGS		{ "audio, ML, inference" };
    MIN_AUTHOR		{ "Valentin Ackva, Fares Schulz, Konstantin Fontaine" };
    MIN_RELATED		{ "nn~" };


    enum MaxType {
        SIGNAL = 0,
        MESSAGE = 1,
        LATENCY = 2,
    };

    struct Input {
        std::unique_ptr<c74::min::inlet<>> inlet;
        MaxType type;
        size_t tensor_index;
        size_t num_channels;
    };

    struct Output {
        std::unique_ptr<c74::min::outlet<>> outlet;
        MaxType type;
        size_t tensor_index;
        size_t num_channels;
    };

    std::vector<Input> m_sig_inlets;
    std::vector<Input> m_msg_inlets;
    std::vector<Output> m_sig_outlets;
    std::vector<Output> m_msg_outlets;

    c74::min::message<> dry_wet;
    c74::min::message<> dspsetup;
    c74::min::message<> anything;
    c74::min::message<> m_float;
    c74::min::message<> m_int;
    c74::min::message<> bang;
    // c74::min::message<> dump;

    bool m_bypass = false;

    void operator()(c74::min::audio_bundle input, c74::min::audio_bundle output) override;
    void parse_input_messages(int inlet_num, const std::vector<float>& args);

private:
    std::string getJsonPath(const c74::min::atoms& args) {
        if (args.size() > 0) {
            std::string input_path = static_cast<std::string>(args[0]);
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

    bool m_valid_config_submitted = false;
    std::string m_config_file_path;

    void init_external(std::vector<size_t> sig_inputs, std::vector<size_t> sig_outputs, std::vector<std::vector<size_t>> msg_inputs, std::vector<std::vector<size_t>> msg_outputs);

    bool initialized = false;
    void prepare(size_t host_buffer_size, double host_sample_rate);
    void prepare_audio_buffers();
    void prepare_latency_outlet(float latency);


    enum class FlowMode {
        Standard,
        RateLocked,
        Adaptive // not yet in use
    };

    struct FlowControl {
        FlowMode mode = FlowMode::Standard;
        size_t samples_accumulated = 0;
    };

    std::vector<FlowControl> m_input_flow_states;

    std::vector<std::vector<float>> m_wet_audio_data;
    std::vector<std::vector<float>> m_dry_audio_data;
    
    std::vector<std::vector<float*>> m_input_channel_ptr;   
    std::vector<std::vector<float*>> m_output_channel_ptr;  
    std::vector<float**> m_input_tensor_ptr;               
    std::vector<float**> m_output_tensor_ptr;              
    
    size_t m_host_buffer_size = 0;
    std::vector<size_t> m_input_sample_counts;
    std::vector<size_t> m_output_sample_counts;
    std::vector<float> m_last_valid_output;

    Mixer m_dry_wet_mixer;
    std::unique_ptr<AniraProcessor> m_anira_processor;

    std::atomic<bool> m_anira_ready_to_process = false;
    std::atomic<bool> m_anira_model_load_confirmed = true;
    bool m_mixing_disabled = false;

    inline const static std::unordered_map<std::string, anira::InferenceBackend> backend_map = {
        {"ONNX", anira::ONNX},
        {"TFLITE", anira::TFLITE},
        {"LIBTORCH", anira::LIBTORCH},
        {"CUSTOM", anira::CUSTOM}
    };
};

#endif //ANIRALIBRARY_TILDE_H
