#include "AniraTilde.h"

// ---- shift from dict to json w/ anira v2 ----
//
// - remove dictionary input
// - get in/out sig/msg infos from json config
// - remove process redundancies from old structure
// - expose flag für message inputs
//
// general: 
// - shift threads-definition to .cpp, only declare in .h
// ---------------------------------------------


AniraTilde::AniraTilde(const c74::min::atoms& args) :
    m_anira_context(get_num_threads()),
    m_audio_config(512, 48000.0),
    dictionary(this, "dictionary", "Provide model configurations via json",
        MIN_FUNCTION {
            c74::min::dict d { args[0] };
            try {
                ModelConfig config = extract_setup_from_dict(d);
                reset_anira();
                setup_anira(config);
            } catch (const std::exception& e) {
                c74::max::error("anira~: %s", e.what());
            }
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
    reset(this, "reset", "Clear current model configuration and reset internal state",
        MIN_FUNCTION {
            try {
                reset_anira();
                c74::max::post("anira~: Model configuration reset successfully");
            } catch (const std::exception& e) {
                c74::max::error("anira~: Error during reset: %s", e.what());
            }
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
    // dump(this, "Dump model config to console",
    //     MIN_FUNCTION {
    //         print_submitted_config();
    //     }
    // )
{
    if (args.size() > 0) {
        m_config_file_path = static_cast<std::string>(args[0]);
        
        if (!m_config_file_path.empty() && std::filesystem::exists(m_config_file_path)) {
            c74::max::post("anira~: Loading config from: %s", m_config_file_path.c_str());
            load_json_config();
        } else {
            c74::max::error("anira~: Config file not found: %s", m_config_file_path.c_str());
        }
    } else {
        c74::max::post("anira~: No config file specified");
    }
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
}

void AniraTilde::init_external(int sig_inputs, int sig_outputs, int msg_inputs, int msg_outputs) {
    m_inlets.clear();
    m_outlets.clear();

    for (int i = 0; i < sig_inputs; ++i) {
        m_inlets.push_back(std::make_unique<c74::min::inlet<>>(this, "(signal) Input " + std::to_string(i + 1), "signal"));
    }

    for (int i = 0; i < sig_outputs; ++i) {
        m_outlets.push_back(std::make_unique<c74::min::outlet<>>(this, "(signal) Output " + std::to_string(i + 1), "signal"));
    }

    for (int i = 0; i < msg_inputs; ++i) {
        m_inlets.push_back(std::make_unique<c74::min::inlet<>>(this, "(message) Input " + std::to_string(i + 1), "float"));
    }

    for (int i = 0; i < msg_outputs; ++i) {
        m_outlets.push_back(std::make_unique<c74::min::outlet<>>(this, "(message) Output " + std::to_string(i + 1), "float"));
    }

    m_outlets.push_back(std::make_unique<c74::min::outlet<>>(this, "(int) Latency Output", "int"));
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

void AniraTilde::load_json_config() {
    if (m_config_file_path.empty() || !std::filesystem::exists(m_config_file_path)) {
        c74::max::error("anira~: Invalid config file path");
        return;
    }

    auto vec_to_str = [this](const std::vector<int64_t>& v){ return vector_to_string(v); };
    auto elem_count = [](const std::vector<int64_t>& shape) -> size_t {
        size_t e = 1;
        for (auto d : shape) e *= static_cast<size_t>(d > 0 ? d : 1);
        return e;
    };

    auto get_vec64 = [](const nlohmann::json& node, const char* key) -> std::vector<int64_t> {
        if (node.contains(key)) return node.at(key).get<std::vector<int64_t>>();
        return {};
    };
    auto get_shapes_from_key = [](const nlohmann::json& node, const char* key) -> std::vector<std::vector<int64_t>> {
        std::vector<std::vector<int64_t>> out;
        if (!node.contains(key) || !node.at(key).is_array()) return out;
        const auto& v = node.at(key);
        // if {single tensor} else {list of tensors}  
        if (!v.empty() && v[0].is_number()) {                       
            out.push_back(v.get<std::vector<int64_t>>());
        } else {                                                    
            for (const auto& t : v) out.push_back(t.get<std::vector<int64_t>>());
        }
        return out;
    };

    try {
        std::ifstream file(m_config_file_path);
        if (!file.is_open()) {
            c74::max::error("anira~: Could not open config file");
            return;
        }
        nlohmann::json j; file >> j; file.close();

        // ---- normalize sources (support old flat keys and new nested keys) ----
        nlohmann::json inf  = j.contains("inference_config") ? j["inference_config"] : j;
        nlohmann::json proc = inf.contains("processing_spec") ? inf["processing_spec"] : j;

        // shapes:
        std::vector<std::vector<int64_t>> inShapes, outShapes;
        if (inf.contains("tensor_shape") && inf["tensor_shape"].is_array() && !inf["tensor_shape"].empty()) {
            const auto& ts = inf["tensor_shape"][0];
            inShapes  = get_shapes_from_key(ts, "input_shape");
            outShapes = get_shapes_from_key(ts, "output_shape");
        } else {
            // fallback
            inShapes  = get_shapes_from_key(j, "input_shape");
            outShapes = get_shapes_from_key(j, "output_shape");
        }

        // processing_spec arrays (per-tensor)
        auto inSz  = get_vec64(proc, "preprocess_input_size");
        auto inCh  = get_vec64(proc, "preprocess_input_channels");
        auto outSz = get_vec64(proc, "postprocess_output_size");
        auto outCh = get_vec64(proc, "postprocess_output_channels");

        // --- debug print ---
        c74::max::post("preprocess input channels: %s", vec_to_str(inCh).c_str());
        c74::max::post("postprocess output channels: %s", vec_to_str(outCh).c_str());
        c74::max::post("preprocess input size: %s", vec_to_str(inSz).c_str());
        c74::max::post("postprocess output size: %s", vec_to_str(outSz).c_str());

        // ---- compute counts ----
        auto safe_vec_at = [](const std::vector<int64_t>& v, size_t i, int64_t def)->int64_t {
            return (i < v.size()) ? v[i] : def;
        };
        auto safe_shape  = [](const std::vector<std::vector<int64_t>>& vv, size_t i)->std::vector<int64_t> {
            if (i < vv.size()) return vv[i];
            return {1}; // default 1 element
        };
        auto vmax3 = [](size_t a, size_t b, size_t c){ return std::max(a, std::max(b, c)); };

        size_t nIn  = vmax3(inSz.size(),  inCh.size(),  inShapes.size());
        size_t nOut = vmax3(outSz.size(), outCh.size(), outShapes.size());

        size_t in_sig = 0, in_msg = 0, out_sig = 0, out_msg = 0;

        for (size_t i = 0; i < nIn; ++i) {
            const int64_t sz = safe_vec_at(inSz, i, 0);
            if (sz > 0) {
                const int64_t ch = std::max<int64_t>(1, safe_vec_at(inCh, i, 1));
                in_sig += static_cast<size_t>(ch);
            } else {
                in_msg += elem_count(safe_shape(inShapes, i));
            }
        }
        for (size_t i = 0; i < nOut; ++i) {
            const int64_t sz = safe_vec_at(outSz, i, 0);
            if (sz > 0) {
                const int64_t ch = std::max<int64_t>(1, safe_vec_at(outCh, i, 1));
                out_sig += static_cast<size_t>(ch);
            } else {
                out_msg += elem_count(safe_shape(outShapes, i));
            }
        }

        // ---- set member vars from new spec ----
        m_num_input_signals   = static_cast<int>(in_sig);
        m_num_input_messages  = static_cast<int>(in_msg);
        m_num_output_signals  = static_cast<int>(out_sig);
        m_num_output_messages = static_cast<int>(out_msg);

        c74::max::post("=> in_sig = %d, in_msg = %d, out_sig = %d, out_msg = %d",
                       m_num_input_signals, m_num_input_messages, m_num_output_signals, m_num_output_messages);

    } catch (const std::exception& e) {
        c74::max::error("anira~: Error parsing JSON: %s", e.what());
    }

    init_external(m_num_input_signals, m_num_output_signals, m_num_input_messages, m_num_output_messages);
}

// void AniraTilde::load_json_config() {
//     if (m_config_file_path.empty() || !std::filesystem::exists(m_config_file_path)) {
//         c74::max::error("anira~: Invalid config file path");
//         return;
//     }

//     try {
//         std::ifstream file(m_config_file_path);
//         if (!file.is_open()) {
//             c74::max::error("anira~: Could not open config file");
//             return;
//         }

//         json j;
//         file >> j;
//         file.close();

//         ModelConfig requested_config;

//         requested_config.model_path = j.value("model_path", "");
//         requested_config.backend = j.value("inference_backend", "");
//         requested_config.max_inference_time = j.value("max_inference_time", 0.0);

//         //udo: first pass in/out shapes to varaiables from arrays
//         //udo: then call reset_anira()
//         //udo: then call setup_anira() with dict

//         requested_config.input_shape = j.contains("input_shape") ? j["input_shape"].get<std::vector<int64_t>>() : std::vector<int64_t>{};
//         requested_config.output_shape = j.contains("output_shape") ? j["output_shape"].get<std::vector<int64_t>>() : std::vector<int64_t>{};
        
//         requested_config.preprocess_input_channels = j.contains("preprocess_input_channels") ? j["preprocess_input_channels"].get<std::vector<int64_t>>() : std::vector<int64_t>{};
//         requested_config.postprocess_output_channels = j.contains("postprocess_output_channels") ? j["postprocess_output_channels"].get<std::vector<int64_t>>() : std::vector<int64_t>{};
//         requested_config.preprocess_input_size = j.contains("preprocess_input_size") ? j["preprocess_input_size"].get<std::vector<int64_t>>() : std::vector<int64_t>{};
//         requested_config.postprocess_output_size = j.contains("postprocess_output_size") ? j["postprocess_output_size"].get<std::vector<int64_t>>() : std::vector<int64_t>{};
        
//         // requested_config.input_signals =
//         // requested_config.input_messages =
//         // requested_config.output_signals =
//         // requested_config.output_messages =

//         // // --- test ----
//         c74::max::post("preprocess input channels: %s", vector_to_string(requested_config.preprocess_input_channels).c_str());
//         c74::max::post("postprocess output channels: %s", vector_to_string(requested_config.postprocess_output_channels).c_str());
//         c74::max::post("preprocess input size: %s", vector_to_string(requested_config.preprocess_input_size).c_str());
//         c74::max::post("postprocess output size: %s", vector_to_string(requested_config.postprocess_output_size).c_str());
//         // -------------

        
//         m_num_input_signals  = j.value("n_input_signals", 0);
//         m_num_input_messages = j.value("n_input_messages", 0);
//         m_num_output_signals = j.value("n_output_signals", 0);
//         m_num_output_messages = j.value("n_output_messages", 0);

//     } catch (const std::exception& e) {
//         c74::max::error("anira~: Error parsing JSON: %s", e.what());
//     }

//     init_external(m_num_input_signals, m_num_output_signals, m_num_input_messages, m_num_output_messages);

//     // return requested_config;
// }

AniraTilde::ModelConfig AniraTilde::extract_setup_from_dict(c74::min::dict& d) {
    ModelConfig requested_config;

    requested_config.model_path = static_cast<std::string>(static_cast<c74::min::atom>(d["modelpath"]));
    requested_config.backend = static_cast<std::string>(static_cast<c74::min::atom>(d["backend"]));
    requested_config.max_inference_time = static_cast<float>(static_cast<c74::min::atom>(d["max_inference_time"]));
    const c74::min::atoms input_shape = d["input_shape"];
    const c74::min::atoms output_shape = d["output_shape"];

    if(!std::filesystem::exists(requested_config.model_path)) {
        throw std::runtime_error("Invalid model path");
    }

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

    const int external_latency_ms = static_cast<int>((external_latency / m_audio_config.m_host_sample_rate) * 1000.0);
    m_outlets.back()->send(external_latency_ms);
}

void AniraTilde::reset_anira() {
    m_anira_ready_to_process.store(false, std::memory_order_release);

    wait_for_model_load();

    if(m_inference_handler != nullptr) {
        m_inference_handler.reset();
    }

    if(m_pp_processor != nullptr) {
        m_pp_processor.reset();
    }

    if(m_inference_config != nullptr) {
        m_inference_config.reset();
    }

    m_dry_audio_data.assign(NUMBER_OF_CHANNELS, std::vector<float>(m_audio_config.m_host_buffer_size, 0.0f));
    m_wet_audio_data.assign(NUMBER_OF_CHANNELS, std::vector<float>(m_audio_config.m_host_buffer_size, 0.0f));

    m_dry_wet_mixer.prepare(
        m_audio_config.m_host_sample_rate,
        m_audio_config.m_host_buffer_size,
        NUMBER_OF_CHANNELS,
        0
    );

    stereo_to_mono = false;

    m_anira_model_load_confirmed.store(true, std::memory_order_release);

    m_outlets.back()->send(0);
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
