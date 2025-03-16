#ifndef ANIRALIBRARY_TILDE_H
#define ANIRALIBRARY_TILDE_H

#include <c74_min.h>
#include <map>
#include <vector>
#include <string>
#include <anira/anira.h>

// Currently only mono or stereo supported
#define NUMBER_OF_CHANNELS 2

class anira_tilde : public c74::min::object<anira_tilde>, public c74::min::vector_operator<> {
public:
    explicit anira_tilde(const c74::min::atoms& args = {});
    static unsigned int get_num_threads();

    MIN_DESCRIPTION { "Max object wrapping the anira inference engine. "
                      "For details visit the <a href='https://github.com/anira-project/anira'>anira repository</a> on Github "
                      "(credits: Valentin Ackva, Fares Schulz)." };
    MIN_TAGS		{ "audio, ML, inference" };
    MIN_AUTHOR		{ "Konstantin Fontaine" };
    MIN_RELATED		{ "nn~" };

    c74::min::inlet<>  input1 { this, "(signal) Input 1", "signal" };
    c74::min::outlet<> output1 { this, "(signal) Output 1", "signal" };

#if NUMBER_OF_CHANNELS == 2
    c74::min::inlet<>  input2 { this, "(signal) Input 2", "signal" };
    c74::min::outlet<> output2 { this, "(signal) Output 2", "signal" };
#endif

    c74::min::argument<c74::min::symbol> model_arg;
    c74::min::argument<c74::min::symbol> backend_arg;

    c74::min::message<> dictionary;
    c74::min::message<> dspsetup;
    c74::min::message<> bang;

    void operator()(c74::min::audio_bundle input, c74::min::audio_bundle output) override;

private:
    void setup_anira();

    struct MaxAudioConfig {
        double samplerate;
        size_t buffer_size;
        size_t num_channels;
    };

    MaxAudioConfig m_max_audio_config {48000, 512, NUMBER_OF_CHANNELS};
    std::vector<std::vector<float>> wet_audio_data;
    std::vector<std::vector<float>> dry_audio_data;

    anira::ContextConfig m_anira_context;
    std::unique_ptr<anira::InferenceConfig> m_inference_config;
    std::unique_ptr<anira::PrePostProcessor> m_pp_processor;
    std::unique_ptr<anira::InferenceHandler> m_inference_handler;
    anira::InferenceBackend m_selected_backend;

    std::string m_model_path;
    std::string m_backend;
    std::vector<int64_t> m_input_shape;
    std::vector<int64_t> m_output_shape;
    int m_external_latency = 0;
    float m_max_inference_time = 0.f;
    bool stereo_to_mono = false;
};

#endif //ANIRALIBRARY_TILDE_H
