#ifndef ANIRALIBRARY_TILDE_H
#define ANIRALIBRARY_TILDE_H

#include <c74_min.h>
#include <map>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include "utils/Mixer.h"
#include "utils/json.hpp"

#include "dsp/AniraProcessor.h"

using json = nlohmann::json;

class AniraTilde : public c74::min::object<AniraTilde>, public c74::min::vector_operator<> {
public:
    explicit AniraTilde(const c74::min::atoms& args = {});
    ~AniraTilde() override;

    MIN_DESCRIPTION { 
        "Neural network inference wrapper for Max. "
        "The anira~ external integrates the <a href='https://github.com/anira-project/anira'>anira</a> library to offer neural network inference inside Max. "
        "It currently supports the following inference engines: LibTorch, ONNXRuntime, and TensorFlow Lite. "
        "At runtime a configuration file can be submitted as dictionary to load a model. " 
    };
    MIN_TAGS		{ "audio, ML, inference" };
    MIN_AUTHOR		{ "Konstantin Fontaine, Valentin Ackva, Fares Schulz" };
    MIN_RELATED		{ "nn~" };


    enum MaxType {
        SIGNAL = 0,
        MESSAGE = 1,
        LATENCY = 2,
    };

    struct Output {
        std::unique_ptr<c74::min::outlet<>> outlet;
        MaxType type;
        size_t anira_index;
    };

    std::vector<std::unique_ptr<c74::min::inlet<>>> m_inlets;
    std::vector<std::unique_ptr<c74::min::outlet<>>> m_outlets;
    std::vector<std::unique_ptr<c74::min::outlet<>>> m_msg_outlets;

    // c74::min::message<> dictionary;
    c74::min::message<> dry_wet;
    c74::min::message<> reset;
    c74::min::message<> dspsetup;
    c74::min::message<> anything;
    c74::min::message<> bang;
    // c74::min::message<> dump;

    void operator()(c74::min::audio_bundle input, c74::min::audio_bundle output) override;

private:
    std::string getJsonPath(const c74::min::atoms& args) {
        if (args.size() > 0) {
            m_config_file_path = static_cast<std::string>(args[0]);
            if (!m_config_file_path.empty() && std::filesystem::exists(m_config_file_path)) {
                c74::max::post("anira~: Loading config from: %s", m_config_file_path.c_str());
                m_valid_config_submitted = true;
                return m_config_file_path;
            } 

            c74::max::error("anira~: Config file not found: %s", m_config_file_path.c_str());
            return "";
        }

        c74::max::post("anira~: No config file specified");
        return "";
    }

    bool m_valid_config_submitted = false;

    std::string m_config_file_path;
    int m_num_input_signals = 0;
    int m_num_input_messages = 0;
    int m_num_output_signals = 0;
    int m_num_output_messages = 0;

    void init_external(int sig_inputs, int sig_outputs, int msg_inputs, int msg_outputs);
    void init_external(std::vector<size_t> sig_inputs, std::vector<size_t> sig_outputs, std::vector<std::vector<size_t>> msg_inputs, std::vector<std::vector<size_t>> msg_outputs);


    void prepare(size_t host_buffer_size, double host_sample_rate);
    void load_json_config();

    static std::string vector_to_string(const std::vector<int64_t>& vec);


    std::vector<std::vector<float>> m_wet_audio_data;
    std::vector<std::vector<float>> m_dry_audio_data;

    Mixer m_dry_wet_mixer;

    std::unique_ptr<AniraProcessor> m_anira_processor;

    std::atomic<bool> m_anira_ready_to_process = false;
    std::atomic<bool> m_anira_model_load_confirmed = true;

    inline const static std::unordered_map<std::string, anira::InferenceBackend> backend_map = {
        {"ONNX", anira::ONNX},
        {"TFLITE", anira::TFLITE},
        {"LIBTORCH", anira::LIBTORCH},
        {"CUSTOM", anira::CUSTOM}
    };
};

#endif //ANIRALIBRARY_TILDE_H
