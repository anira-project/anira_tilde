#include "AniraTilde.h"

// ---- shift from dict to json w/ anira v2 ----
//
// - cleanup
// - generalize set / get non-streamable inlets (the anything function currently only works for gain example)
// - fix hardcoded channel input in process block 
// x shift outlets / inlets into struct (clean indexing)
// - connect latency outlet
// - latent parameter access / compression (sample value assignment)
// - AniraProcessor.cpp 
// - check we meet the num_inlets when max is pasing a message

// general: 
// - shift threads-definition to .cpp, only declare in .h
// - add dump msg
// - avoid redundant double / float conversion 
// - expose flag für message inputs
// - mute / bypass
// ---------------------------------------------


AniraTilde::AniraTilde(const c74::min::atoms& args) :
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
                // reset_anira();
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
            prepare(buffer_size, sample_rate);
            initialized = true;
            return {};
        }
    ),
    bang(this, "bang", "Trigger processing",
        MIN_FUNCTION {
            // for(int i = 0; i < m_msg_outlets.size(); ++i){
            //     m_msg_outlets[i]->send(m_anira_processor->get_output(1, 0));
            // } 
            if (!initialized || m_anira_processor == nullptr) {
                c74::max::error("anira~: External not initialized. Activate DSP.");
                return {};
            }

            for(int i = 0; i < m_msg_outlets.size(); ++i){
                if (m_msg_outlets[i].type != MaxType::MESSAGE) {
                    continue;
                }
                size_t tensor_index = m_msg_outlets[i].tensor_index;
                size_t num_channels = m_msg_outlets[i].num_channels;
                std::vector<float> msg_data;
                for(size_t j = 0; j < num_channels; ++j){
                    msg_data.push_back(m_anira_processor->get_output(tensor_index, j));
                }
                if(num_channels == 1){
                    m_msg_outlets[i].outlet->send(msg_data[0]);
                } else {
                    c74::min::atoms msg_atoms;
                    msg_atoms.push_back("list");
                    for(const auto& val : msg_data){
                        msg_atoms.push_back(val);
                    }
                    m_msg_outlets[i].outlet->send(msg_atoms);
                }
            }

            for(int i = 0; i < m_msg_outlets.size(); ++i){
                if (m_msg_outlets[i].type == MaxType::LATENCY) {
                    m_msg_outlets[i].outlet->send(static_cast<int>(m_anira_processor->get_latency_samples()));
                }
            }
            return {};
        }
    ),
    anything(this, "anything", "Receive lists/floats in message inlets",
        MIN_FUNCTION {
            const int inlet_num = inlet;
            
            c74::max::post("anira~: Received data on inlet %d", inlet_num);
            const size_t num_sig_inputs = m_sig_inlets.size();
            const size_t msg_index = inlet_num - num_sig_inputs; 
            
            std::vector<float> msg_data;
            if (m_msg_inlets[msg_index].num_channels != args.size() - 1) {
                c74::max::error("anira~: Incorrect number of elements for inlet %zu. Expected %zu but got %zu.", msg_index + 1, m_msg_inlets[msg_index].num_channels, args.size() - 1);
                return {};
            }

            for (size_t i = 0; i < args.size(); ++i) {
                if (i == 0){
                    if(std::string(args[i]) != "list"){
                        c74::max::error("anira~: Input not of type 'list', received: %s", std::string(args[i]).c_str());
                        return {};
                    }
                } else {
                    msg_data.push_back(static_cast<float>(args[i]));
                }
            }

            for(size_t i = 0; i < msg_data.size(); ++i){
                m_anira_processor->set_input(msg_data[i], m_msg_inlets[msg_index].tensor_index, i);
            }
            return {};
        }
    )

    // dump(this, "Dump model config to console",
    //     MIN_FUNCTION {
    //         print_submitted_config();
    //     }
    // )
{
    getJsonPath(args);

    if (m_valid_config_submitted) {
        load_json_config();
        m_anira_processor = std::make_unique<AniraProcessor>(m_config_file_path);

        c74::max::post("---> inSigCh: %zu", m_anira_processor->inSigCh.size());
        c74::max::post("---> outSigCh: %zu", m_anira_processor->outSigCh.size());
        c74::max::post("---> inMsgCh: %zu", m_anira_processor->inMsgCh.size());
        c74::max::post("---> outMsgCh: %zu", m_anira_processor->outMsgCh.size());
        init_external(m_anira_processor->inSigCh, m_anira_processor->outSigCh, m_anira_processor->inMsgCh, m_anira_processor->outMsgCh);
    }     
}

AniraTilde::~AniraTilde() {
}

void AniraTilde::operator()(c74::min::audio_bundle input, c74::min::audio_bundle output) {
    const auto num_channels_print = static_cast<size_t>(input.channel_count());
    c74::max::post("posting number of input channels: %zu", num_channels_print);
    
    const auto num_channels = 1;
    const auto sample_count = static_cast<size_t>(input.frame_count());
    
    for (int channel = 0; channel < num_channels; ++channel) {
        for (int sample = 0; sample < sample_count; ++sample) {
            auto value = static_cast<float>(input.samples(channel)[sample]);
            m_dry_audio_data[channel][sample] = value;
        }
    }

    float* input_data_ptr = m_dry_audio_data.data()->data();
    float* output_data_ptr = m_wet_audio_data.data()->data();
    const bool anira_initialized = m_anira_ready_to_process.load(std::memory_order_acquire);

    if (m_anira_processor)
    {
        float* inputs[1]  = { input_data_ptr };
        float* outputs[1] = { output_data_ptr };

        m_anira_processor->process(inputs, outputs, sample_count);
    }

    // if(m_inference_handler != nullptr && anira_initialized){
    //     // m_inference_handler->process(&input_data_ptr, &output_data_ptr, sample_count);
    // }

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

// void AniraTilde::init_external(int sig_inputs, int sig_outputs, int msg_inputs, int msg_outputs) {
//     m_inlets.clear();
//     m_outlets.clear();

//     for (int i = 0; i < sig_inputs; ++i) {
//         m_inlets.push_back(std::make_unique<c74::min::inlet<>>(this, "(signal) Input " + std::to_string(i + 1), "signal"));
//     }

//     for (int i = 0; i < sig_outputs; ++i) {
//         m_outlets.push_back(std::make_unique<c74::min::outlet<>>(this, "(signal) Output " + std::to_string(i + 1), "signal"));
//     }

//     for (int i = 0; i < msg_inputs; ++i) {
//         m_inlets.push_back(std::make_unique<c74::min::inlet<>>(this, "(message) Input " + std::to_string(i + 1), "float"));
//     }

//     for (int i = 0; i < msg_outputs; ++i) {
//         m_outlets.push_back(std::make_unique<c74::min::outlet<>>(this, "(message) Output " + std::to_string(i + 1), "float"));
//     }

//     m_outlets.push_back(std::make_unique<c74::min::outlet<>>(this, "(int) Latency Output", "int"));
// }

void AniraTilde::init_external(std::vector<size_t> sig_inputs, std::vector<size_t> sig_outputs, std::vector<std::vector<size_t>> msg_inputs, std::vector<std::vector<size_t>> msg_outputs) {
    m_sig_inlets.clear();
    m_msg_inlets.clear();
    m_sig_outlets.clear();
    m_msg_outlets.clear();

    size_t last_audio_input_tensor = 0;
    size_t last_audio_output_tensor = 0;

    for (int i = 0; i < sig_inputs.size(); ++i) {
        last_audio_input_tensor++;
        for (int j = 0; j < sig_inputs[i]; ++j) {
            Input input;
            input.inlet = std::make_unique<c74::min::inlet<>>(this, "(signal) Tensor " + std::to_string(i + 1) + ", Channel " + std::to_string(j + 1), "signal");
            input.type = MaxType::SIGNAL;
            input.num_channels = 1;
            input.tensor_index = i;
            m_sig_inlets.push_back(std::move(input));
        }
    }

    for (int i = 0; i < sig_outputs.size(); ++i) {
        last_audio_output_tensor++;
        for (int j = 0; j < sig_outputs[i]; ++j) {
            Output output; 
            output.outlet = std::make_unique<c74::min::outlet<>>(this, "(signal) Tensor " + std::to_string(i + 1) + ", Channel " + std::to_string(j + 1), "signal");
            output.type = MaxType::SIGNAL;
            output.num_channels = 1;
            output.tensor_index = i;
            m_sig_outlets.push_back(std::move(output));
        }
    }


    for (int tensor = 0; tensor < msg_inputs.size(); ++tensor) {
        for (int channel = 0; channel < msg_inputs[tensor].size(); ++channel) {
            size_t num_inlets = msg_inputs[tensor][channel];
            Input input; 
            // std::string type = (num_inlets <= 1) ? "float" : "list";
            std::string type = (num_inlets <= 1) ? "list" : "list";
            input.inlet = std::make_unique<c74::min::inlet<>>(this, "(message) Tensor " + std::to_string(tensor + 1) + ", Channel " + std::to_string(channel + 1), type);
            input.type = MaxType::MESSAGE;
            input.num_channels = num_inlets;
            input.tensor_index = tensor + last_audio_input_tensor;
            m_msg_inlets.push_back(std::move(input));
        }
    }

    for (int tensor = 0; tensor < msg_outputs.size(); ++tensor) {
        for (int channel = 0; channel < msg_outputs[tensor].size(); ++channel) {
            size_t num_outlets = msg_outputs[tensor][channel];
            Output output; 
            // std::string type = (num_outlets <= 1) ? "float" : "list";
            std::string type = (num_outlets <= 1) ? "list" : "list";
            output.outlet = std::make_unique<c74::min::outlet<>>(this, "(message) Tensor " + std::to_string(tensor + 1) + ", Channel " + std::to_string(channel + 1), type);
            output.type = MaxType::MESSAGE;
            output.num_channels = num_outlets;
            output.tensor_index = tensor + last_audio_output_tensor;
            m_msg_outlets.push_back(std::move(output));
        }
    }

    Output output;
    output.outlet = std::make_unique<c74::min::outlet<>>(this, "(int) Latency Output", "int");
    output.type = MaxType::LATENCY;
    output.num_channels = 1;
    m_msg_outlets.push_back(std::move(output));
}


void AniraTilde::prepare(size_t host_buffer_size, double host_sample_rate) {
    m_dry_audio_data.assign(2, std::vector<float>(host_buffer_size, 0.0f));
    m_wet_audio_data.assign(2, std::vector<float>(host_buffer_size, 0.0f));

    float latency = 0.f;
    
    if(m_anira_processor) {
        m_anira_processor->prepare(host_buffer_size, host_sample_rate);
        latency = static_cast<float>(m_anira_processor->get_latency_samples());
        for(int i = 0; i < m_msg_outlets.size(); ++i){
            if(m_msg_outlets[i].type == MaxType::LATENCY){
                m_msg_outlets[i].outlet->send(static_cast<int>(latency));
            }
        }
    }

    m_dry_wet_mixer.prepare(
        host_sample_rate,
        host_buffer_size,
        2,
        latency
    );
}

void AniraTilde::load_json_config() {
    return;


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

// void AniraTilde::reset_anira() {
//     m_anira_ready_to_process.store(false, std::memory_order_release);

//     wait_for_model_load();

//     if(m_inference_handler != nullptr) {
//         m_inference_handler.reset();
//     }

//     if(m_pp_processor != nullptr) {
//         m_pp_processor.reset();
//     }

//     if(m_inference_config != nullptr) {
//         m_inference_config.reset();
//     }

//     m_dry_audio_data.assign(NUMBER_OF_CHANNELS, std::vector<float>(m_audio_config.m_buffer_size, 0.0f));
//     m_wet_audio_data.assign(NUMBER_OF_CHANNELS, std::vector<float>(m_audio_config.m_buffer_size, 0.0f));

//     m_dry_wet_mixer.prepare(
//         m_audio_config.m_sample_rate,
//         m_audio_config.m_buffer_size,
//         NUMBER_OF_CHANNELS,
//         0
//     );

//     m_anira_model_load_confirmed.store(true, std::memory_order_release);

//     m_outlets.back()->send(0);
// }

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

std::string AniraTilde::vector_to_string(const std::vector<size_t>& vec) {
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
