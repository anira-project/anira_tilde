#ifndef ANIRALIBRARY_TILDE_H
#define ANIRALIBRARY_TILDE_H

#include <c74_min.h>
#include <map>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <anira/anira.h>
#include "utils/Mixer.h"
#include "utils/json.hpp"
using json = nlohmann::json;

// Currently only mono or stereo supported
#define NUMBER_OF_CHANNELS 2

class AniraTilde : public c74::min::object<AniraTilde>, public c74::min::vector_operator<> {
public:
    explicit AniraTilde(const c74::min::atoms& args = {});
    ~AniraTilde() override;

    static unsigned int get_num_threads();

    MIN_DESCRIPTION { 
        "Neural network inference wrapper for Max. "
        "The anira~ external integrates the <a href='https://github.com/anira-project/anira'>anira</a> library to offer neural network inference inside Max. "
        "It currently supports the following inference engines: LibTorch, ONNXRuntime, and TensorFlow Lite. "
        "At runtime a configuration file can be submitted as dictionary to load a model. " 
    };
    MIN_TAGS		{ "audio, ML, inference" };
    MIN_AUTHOR		{ "Konstantin Fontaine, Valentin Ackva, Fares Schulz" };
    MIN_RELATED		{ "nn~" };

    std::vector<std::unique_ptr<c74::min::inlet<>>> m_inlets;
    std::vector<std::unique_ptr<c74::min::outlet<>>> m_outlets;

    c74::min::message<> dictionary;
    c74::min::message<> dry_wet;
    c74::min::message<> reset;
    c74::min::message<> dspsetup;
    // c74::min::message<> dump;

    c74::min::attribute<int> threads {
        this, "threads", 0,
        c74::min::description{"Number of audio threads"},
        c74::min::range{0, 64},
        c74::min::setter{ MIN_FUNCTION {
            return args;
        }}
    };

    void operator()(c74::min::audio_bundle input, c74::min::audio_bundle output) override;

private:
    struct ModelConfig {
        std::string model_path;
        std::string backend;
        std::vector<int64_t> input_shape;
        std::vector<int64_t> output_shape;
        std::vector<int64_t> preprocess_input_channels;
        std::vector<int64_t> postprocess_output_channels;
        std::vector<int64_t> preprocess_input_size;
        std::vector<int64_t> postprocess_output_size;
        float max_inference_time;
    };

    bool m_threads_set = false;

    std::string m_config_file_path;
    int m_num_input_signals = 0;
    int m_num_input_messages = 0;
    int m_num_output_signals = 0;
    int m_num_output_messages = 0;

    void init_external(int sig_inputs, int sig_outputs, int msg_inputs, int msg_outputs);
    void prepare(size_t host_buffer_size, double host_sample_rate);
    void load_json_config(); // load json file directly 
    static ModelConfig extract_setup_from_dict(c74::min::dict& d); // load a dict instead of json
    void setup_anira(ModelConfig& config);
    void reset_anira();
    void wait_for_model_load();

    static void print_submitted_config(ModelConfig& config);
    static std::string vector_to_string(const std::vector<int64_t>& vec);


    std::vector<std::vector<float>> m_wet_audio_data;
    std::vector<std::vector<float>> m_dry_audio_data;

    anira::HostAudioConfig m_audio_config;
    anira::ContextConfig m_anira_context;
    anira::InferenceBackend m_selected_backend;
    std::unique_ptr<anira::InferenceConfig> m_inference_config;
    std::unique_ptr<anira::PrePostProcessor> m_pp_processor;
    std::unique_ptr<anira::InferenceHandler> m_inference_handler;

    bool stereo_to_mono = false;
    Mixer m_dry_wet_mixer;

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
