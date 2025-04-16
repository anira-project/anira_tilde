#include "AniraTilde.h"

AniraTilde::AniraTilde(const c74::min::atoms& args) :
    m_anira_context(get_num_threads()),
    m_audio_config(512, 48000.0),
    dictionary(this, "dictionary", "Provide model configurations via json",
        MIN_FUNCTION {
            c74::min::dict d { args[0] };
            ModelConfig config = extract_setup_from_dict(d);
            setup_anira(config);
            return {};
        }
    ),
    dry_wet(this, "mix", "Set the dry/wet mix of the output",
        MIN_FUNCTION {
            const float new_mix = std::clamp(static_cast<float>(args[0]), 0.0f, 100.0f) / 100.0f;
            m_dry_wet_mixer.set_mix(new_mix);
            return {};
        }
    ),
    dspsetup(this, "dspsetup",
        MIN_FUNCTION {
            const auto sample_rate = static_cast<double>(args[0]);
            const auto buffer_size = static_cast<size_t>(args[1]);
            const int m_threads = (threads > 0 ? static_cast<int>(threads) : get_num_threads());
            m_anira_context = anira::ContextConfig(m_threads);
            
            prepare(buffer_size, sample_rate);
            return {};
        }
    )
{
}

AniraTilde::~AniraTilde() {
}

unsigned int AniraTilde::get_num_threads() {
    return std::thread::hardware_concurrency() / 2 > 0 ? std::thread::hardware_concurrency() / 2 : 1;
}

void AniraTilde::operator()(c74::min::audio_bundle input, c74::min::audio_bundle output) {
    const auto num_channels = static_cast<size_t>(input.channel_count());
    const auto sample_count = static_cast<size_t>(input.frame_count());

    if (stereo_to_mono) {
        for (int channel = 0; channel < num_channels; ++channel) {
            for (int sample = 0; sample < sample_count; ++sample) {
                auto valueLeft = static_cast<float>(input.samples(0)[sample] * 0.5f);
                auto valueRight = static_cast<float>(input.samples(1)[sample] * 0.5f);
                m_dry_audio_data[channel][sample] = valueLeft + valueRight;
            }
        }
    } else {
        for (int channel = 0; channel < num_channels; ++channel) {
            for (int sample = 0; sample < sample_count; ++sample) {
                auto value = static_cast<float>(input.samples(channel)[sample]);
                m_dry_audio_data[channel][sample] = value;
            }
        }
    }

    float* input_data_ptr = m_dry_audio_data.data()->data();
    float* output_data_ptr = m_wet_audio_data.data()->data();
    const bool anira_initialized = m_anira_ready_to_process.load(std::memory_order_acquire);

    if(m_inference_handler != nullptr && anira_initialized){
        m_inference_handler->process(&input_data_ptr, &output_data_ptr, sample_count);
    }

    if (stereo_to_mono) {
        for (int sample = 0; sample < sample_count; ++sample) {
            m_wet_audio_data[1][sample] = m_wet_audio_data[0][sample];
        }
    }

    for (int channel = 0; channel < num_channels; ++channel) {
        for (int sample = 0; sample < sample_count; ++sample) {
            if (anira_initialized) {
                m_dry_wet_mixer.push_dry_sample(m_dry_audio_data[channel][sample], channel);
                float value = m_dry_wet_mixer.mix_wet_sample(m_wet_audio_data[channel][sample], channel);
                output.samples(channel)[sample] = static_cast<double>(value);
            } else {
                float value = m_dry_audio_data[channel][sample];
                output.samples(channel)[sample] = static_cast<double>(value);
            }
        }
    }

    if (m_anira_model_load_confirmed.load(std::memory_order_acquire) == false) {
        m_anira_model_load_confirmed.store(true, std::memory_order_release);
    }

    // if(m_inference_handler != nullptr && anira_initialized){
    //     c74::max::post("Mixer latency: %d", m_dry_wet_mixer.get_latency());
    //     c74::max::post("Inference latency: %d", m_inference_handler->get_latency());
    // }
}

void AniraTilde::prepare(size_t host_buffer_size, double host_sample_rate) {
    m_audio_config.m_host_sample_rate = host_sample_rate;
    m_audio_config.m_host_buffer_size = host_buffer_size;

    m_dry_audio_data.assign(NUMBER_OF_CHANNELS, std::vector<float>(m_audio_config.m_host_buffer_size, 0.0f));
    m_wet_audio_data.assign(NUMBER_OF_CHANNELS, std::vector<float>(m_audio_config.m_host_buffer_size, 0.0f));

    if (m_inference_handler != nullptr){
        m_inference_handler->prepare(m_audio_config);

        bool mono_model = (m_inference_config->m_num_audio_channels[anira::Input] == 1 &&
                           m_inference_config->m_num_audio_channels[anira::Output] == 1);
        stereo_to_mono = (mono_model && NUMBER_OF_CHANNELS == 2);

        m_dry_wet_mixer.prepare(
            m_audio_config.m_host_sample_rate,
            m_audio_config.m_host_buffer_size,
            NUMBER_OF_CHANNELS,
            m_inference_handler->get_latency()
        );
    } else {
        stereo_to_mono = false;
        m_dry_wet_mixer.prepare(
            m_audio_config.m_host_sample_rate,
            m_audio_config.m_host_buffer_size,
            NUMBER_OF_CHANNELS,
            0
        );
    }
}

AniraTilde::ModelConfig AniraTilde::extract_setup_from_dict(c74::min::dict& d) {
    ModelConfig requested_config;

    requested_config.model_path = static_cast<std::string>(static_cast<c74::min::atom>(d["modelpath"]));
    requested_config.backend = static_cast<std::string>(static_cast<c74::min::atom>(d["backend"]));
    requested_config.max_inference_time = static_cast<float>(static_cast<c74::min::atom>(d["max_inference_time"]));
    const c74::min::atoms input_shape = d["input_shape"];
    const c74::min::atoms output_shape = d["output_shape"];

    for (auto i : input_shape) {
        requested_config.input_shape.emplace_back(static_cast<int64_t>(i));
    }

    for (auto i : output_shape) {
        requested_config.output_shape.emplace_back(static_cast<int64_t>(i));
    }

    return requested_config;
}


void AniraTilde::setup_anira(ModelConfig& config) {
    if (config.model_path.empty() || config.backend.empty() || config.input_shape.empty() ||
        config.output_shape.empty() || config.max_inference_time == 0.f) {
        c74::max::error("anira~: Invalid configuration, please provide all necessary parameters");
        return;
    }

    if (auto backend = backend_map.find(config.backend); backend != backend_map.end()) {
        m_selected_backend = backend->second;
    } else {
        c74::max::error("anira~: Invalid backend: %s", config.backend.c_str());
    }

    m_anira_ready_to_process.store(false, std::memory_order_release);
    wait_for_model_load();

    std::vector<anira::ModelData> model_config = {{config.model_path, m_selected_backend}};
    std::vector<anira::TensorShape> tensor_shape = {{
        {config.input_shape},
        {config.output_shape},
        m_selected_backend
    }};

    unsigned int model_latency = 0;
    unsigned int number_of_warm_up_runs = 0;

    m_inference_config = std::make_unique<anira::InferenceConfig>(
        model_config,
        tensor_shape,
        config.max_inference_time,
        model_latency,
        number_of_warm_up_runs
    );

    // Construct custom anira::PrePostProcessor
    // m_pp_processor = std::make_unique<CNNPrePostProcessor>(*m_inference_config);

    // If no custom anira::PrePostProcessor is needed, the default anira::PrePostProcessor can be used
    m_pp_processor = std::make_unique<anira::PrePostProcessor>(*m_inference_config);

    m_inference_handler = std::make_unique<anira::InferenceHandler>(*m_pp_processor, *m_inference_config);

    m_inference_handler->prepare(m_audio_config);

    const int external_latency = m_inference_handler->get_latency();
    m_dry_wet_mixer.prepare(
        m_audio_config.m_host_sample_rate,
        m_audio_config.m_host_buffer_size,
        NUMBER_OF_CHANNELS,
        external_latency
    );

    const bool mono_model = (m_inference_config->m_num_audio_channels[anira::Input] == 1 &&
                             m_inference_config->m_num_audio_channels[anira::Output] == 1);

    stereo_to_mono = (mono_model && NUMBER_OF_CHANNELS == 2);

    m_inference_handler->set_inference_backend(m_selected_backend);

    m_anira_ready_to_process.store(true, std::memory_order_release);

    const int external_latency_ms = external_latency / m_audio_config.m_host_sample_rate * 1000;
    latency_output.send(external_latency_ms);
}

// TODO: Check if we can suspend processing instead of this busy waiting
void AniraTilde::wait_for_model_load() {
    // Check if audio is running and wait for confirmation to safely initialize anira
    if (c74::max::sys_getdspobjdspstate(*this) == 1) {
        m_anira_model_load_confirmed.store(false, std::memory_order_release);

        constexpr int sleep_time_ms = 5;

        const auto buffer_size = static_cast<float>(m_audio_config.m_host_buffer_size);
        const auto sample_rate = static_cast<float>(m_audio_config.m_host_sample_rate);
        const auto buffer_size_ms = (buffer_size / sample_rate) * 1000.0f;
        const auto max_counter = static_cast<int>(buffer_size_ms / sleep_time_ms) * 3;

        for (int counter = 0; counter <= std::max(500/sleep_time_ms, max_counter); ++counter) {
            if (m_anira_model_load_confirmed.load(std::memory_order_acquire)) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_ms));
        }
        c74::max::error("anira~: Model load confirmation timeout");
    }
}

void AniraTilde::print_submitted_config(ModelConfig& config) {
    c74::max::post("anira~: -- Request new anira setup --");
    c74::max::post("anira~: Model path: %s", config.model_path.c_str());
    c74::max::post("anira~: Selected backend: %s", config.backend.c_str());
    c74::max::post("anira~: Input shape: %s", vector_to_string(config.input_shape).c_str());
    c74::max::post("anira~: Output shape: %s", vector_to_string(config.output_shape).c_str());
    c74::max::post("anira~: Maximum inference time: %.1fms", config.max_inference_time);
}

std::string AniraTilde::vector_to_string(const std::vector<int64_t>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i < vec.size() - 1)
            oss << ", ";
    }
    oss << "]";
    return oss.str();
}

MIN_EXTERNAL_CUSTOM(AniraTilde, anira~);
