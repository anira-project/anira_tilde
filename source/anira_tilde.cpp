// external wrapping the anira library
// c Konstantin Fontaine

#include "anira_tilde.h"

anira_tilde::anira_tilde(const c74::min::atoms& args) : m_anira_context(get_num_threads()),
    model_arg(this, "model_path", "Initial model path.",
        MIN_ARGUMENT_FUNCTION {
            std::string model_path = arg;
            m_model_path = model_path;
            std::cout << "initial model path: " << m_model_path << std::endl;
        }
    ),
    backend_arg(this, "model_backend", "Initial model backend.",
        MIN_ARGUMENT_FUNCTION {
            std::string model_backend = arg;
            m_backend = model_backend;
            std::cout << "initial model backend: " << m_backend << std::endl;
        }
    ),
    dictionary(this, "dictionary", "Provide model configurations via json",
        MIN_FUNCTION {
            c74::min::dict d { args[0] };

            m_model_path = static_cast<c74::min::atom>(d["modelpath"]);
            m_backend = static_cast<c74::min::atom>(d["backend"]);
            m_max_inference_time = static_cast<c74::min::atom>(d["max_inference_time"]);
            const c74::min::atoms input_shape = d["input_shape"];
            const c74::min::atoms output_shape = d["output_shape"];

            m_input_shape.clear();
            m_output_shape.clear();

            for (auto i : input_shape) {
                m_input_shape.emplace_back(static_cast<int64_t>(i));
            }

            for (auto i : output_shape) {
                m_output_shape.emplace_back(static_cast<int64_t>(i));
            }

            setup_anira();

            return {};
        }
    ),
    dspsetup(this, "dspsetup",
        MIN_FUNCTION {
            m_max_audio_config.samplerate = static_cast<double>(args[0]);
            m_max_audio_config.buffer_size = static_cast<size_t>(args[1]);

            dry_audio_data.clear();
            for (int i = 0; i < m_max_audio_config.num_channels; ++i) {
                dry_audio_data.emplace_back(m_max_audio_config.buffer_size, 0.0f);
            }

            wet_audio_data.clear();
            for (int i = 0; i < m_max_audio_config.num_channels; ++i) {
                wet_audio_data.emplace_back(m_max_audio_config.buffer_size, 0.0f);
            }

            if (m_inference_handler != nullptr){
                anira::HostAudioConfig host_audio_config = {m_max_audio_config.buffer_size, m_max_audio_config.samplerate};
                m_inference_handler->prepare(host_audio_config);
                m_external_latency = m_inference_handler->get_latency();

                bool mono_model = (m_inference_config->m_num_audio_channels[anira::Input] == 1 && m_inference_config->m_num_audio_channels[anira::Output] == 1);

                if (mono_model && m_max_audio_config.num_channels == 2) {
                    stereo_to_mono = true;
                } else {
                    stereo_to_mono = false;
                }
            }

            return {};
        }
    ),
    bang(this, "bang", "Dump all variables to console.",
        MIN_FUNCTION {
            std::cout << "sample rate: " << m_max_audio_config.samplerate << std::endl;
            std::cout << "buffer size: " << m_max_audio_config.buffer_size << std::endl;
            return {};
        }
    )
{
}

unsigned int anira_tilde::get_num_threads() {
    return std::thread::hardware_concurrency() / 2 > 0 ? std::thread::hardware_concurrency() / 2 : 1;
}

void anira_tilde::operator()(c74::min::audio_bundle input, c74::min::audio_bundle output) {
    const size_t num_channels = static_cast<size_t>(input.channel_count());
    const size_t sample_count = static_cast<size_t>(input.frame_count());

    if (stereo_to_mono) {
        for (int i = 0; i < sample_count; ++i) {
            auto valueLeft = static_cast<float>(input.samples(0)[i] * 0.5f);
            auto valueRight = static_cast<float>(input.samples(1)[i] * 0.5f);
            dry_audio_data[0][i] = valueLeft + valueRight;
        }
    } else {
        for (int channel = 0; channel < num_channels; ++channel) {
            for (int i = 0; i < sample_count; ++i) {
                auto value = static_cast<float>(input.samples(channel)[i]);
                dry_audio_data[channel][i] = value;
            }
        }
    }

    float* input_data_ptr = dry_audio_data.data()->data();
    float* output_data_ptr = wet_audio_data.data()->data();

    if(m_inference_handler != nullptr){
        m_inference_handler->process(&input_data_ptr, &output_data_ptr, sample_count);
    }

    if (stereo_to_mono) {
        for (int i = 0; i < sample_count; ++i) {
            wet_audio_data[1][i] = wet_audio_data[0][i];
        }
    }

    for (int channel = 0; channel < num_channels; ++channel) {
        for (int i = 0; i < sample_count; ++i) {
            auto value = static_cast<double>(wet_audio_data[channel][i]);
            output.samples(channel)[i] = value;
        }
    }
}


void anira_tilde::setup_anira() {
    std::cout << "-- Request new anira setup --" << std::endl;
    std::cout << "Model path: " << m_model_path << std::endl;
    std::cout << "Selected backend: " << m_backend << std::endl;
    std::cout << "Input shape: " << m_input_shape << std::endl;
    std::cout << "Output shape: " << m_output_shape << std::endl;
    std::cout << "Maximum inference time: " << m_max_inference_time << "ms" << std::endl;

    if(m_backend == "ONNX") {
        m_selected_backend = anira::ONNX;
    } else if (m_backend == "TFLITE") {
        m_selected_backend = anira::TFLITE;
    } else if (m_backend == "LIBTORCH") {
        m_selected_backend = anira::LIBTORCH;
    } else {
        std::cerr << "Invalid backend: " << m_backend << std::endl;
        return;
    }

    // Configure neural network model and corresponding inference backend
    std::vector<anira::ModelData> model_config;
    model_config.emplace_back(m_model_path, m_selected_backend);

    // Configure input and output shapes
    std::vector<anira::TensorShape> tensor_shape = {{{m_input_shape}, {m_output_shape}, m_selected_backend}};

    // If the model has a fixed latency, it can be set here
    unsigned int model_latency = 0;

    // To speed up the inference process, the model can be warmed up with a number of runs
    unsigned int number_of_warm_up_runs = 10;

    // Define maximum inference time
    m_inference_config = std::make_unique<anira::InferenceConfig>(model_config, tensor_shape, m_max_inference_time, model_latency, number_of_warm_up_runs);

    // Construct custom anira::PrePostProcessor
    // m_pp_processor = std::make_unique<CNNPrePostProcessor>(*m_inference_config);

    // If no custom anira::PrePostProcessor is needed, the default anira::PrePostProcessor can be used
    m_pp_processor = std::make_unique<anira::PrePostProcessor>(*m_inference_config);

    // Construct anira::InferenceHandler and submit host audio configuration
    m_inference_handler = std::make_unique<anira::InferenceHandler>(*m_pp_processor, *m_inference_config);
    anira::HostAudioConfig host_audio_config = {m_max_audio_config.buffer_size, m_max_audio_config.samplerate};
    m_inference_handler->prepare(host_audio_config);

    // Get the latency of the inference handler to delay the dry signal
    m_external_latency = m_inference_handler->get_latency();

    bool mono_model = (m_inference_config->m_num_audio_channels[anira::Input] == 1 && m_inference_config->m_num_audio_channels[anira::Output] == 1);

    if (mono_model && m_max_audio_config.num_channels == 2) {
        stereo_to_mono = true;
    } else {
        stereo_to_mono = false;
    }

    // Select the backend for inference
    m_inference_handler->set_inference_backend(m_selected_backend);
}

MIN_EXTERNAL(anira_tilde);
