#include "AniraTilde.h"

AniraTilde::AniraTilde(const c74::min::atoms& args) :
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
            prepare(buffer_size, sample_rate);
            initialized = true;
            return {};
        }
    ),
    bang(this, "bang", "Output non-streamable parameters",
        MIN_FUNCTION {
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
    anything(this, "list", "Receive lists in message inlets",
        MIN_FUNCTION {
            const int inlet_num = inlet;
            std::vector<float> msg_data;
            msg_data.reserve(args.size()); 

            for(size_t i = 0; i < args.size(); ++i) {
                msg_data.push_back(static_cast<float>(args[i]));
            }

            parse_input_messages(inlet_num, msg_data);
            return {};
        }
    ),
    m_float(this, "float", "Receive float in message inlets",
        MIN_FUNCTION {
            const int inlet_num = inlet;
            std::vector<float> msg_data = { static_cast<float>(args[0]) };
            parse_input_messages(inlet_num, msg_data);
            return {};
        }
    ),
    m_int(this, "int", "Receive int in message inlets",
        MIN_FUNCTION {
            const int inlet_num = inlet;
            std::vector<float> msg_data = { static_cast<float>(args[0]) };
            parse_input_messages(inlet_num, msg_data);
            return {};
        }
    )
{
    getJsonPath(args);

    if (m_valid_config_submitted) {
        m_anira_processor = std::make_unique<AniraProcessor>(m_config_file_path);

        c74::max::post(
            "anira~: Signal input tensors: %zu, Signal output tensors: %zu, Message input tensors: %zu, Message output tensors: %zu",
            m_anira_processor->inSigCh.size(),
            m_anira_processor->outSigCh.size(),
            m_anira_processor->inMsgCh.size(),
            m_anira_processor->outMsgCh.size()
        );
        init_external(m_anira_processor->inSigCh, m_anira_processor->outSigCh, m_anira_processor->inMsgCh, m_anira_processor->outMsgCh);
    }     
}

AniraTilde::~AniraTilde() {
}

void AniraTilde::operator()(c74::min::audio_bundle input, c74::min::audio_bundle output) {
    const auto num_channels = static_cast<size_t>(m_sig_inlets.size());
    const auto sample_count = static_cast<size_t>(input.frame_count());
    
    const bool anira_initialized = m_anira_ready_to_process.load(std::memory_order_acquire);

    for (size_t channel = 0; channel < num_channels; ++channel) {
        for (size_t sample = 0; sample < sample_count; ++sample) {
            m_dry_audio_data[channel][sample] = static_cast<float>(input.samples(channel)[sample]);
        }
    }

    if (m_anira_processor && anira_initialized)
    {
        const float* const* const* input_data = const_cast<const float* const* const*>(m_input_tensor_ptr.data());
        float* const* const* output_data = m_output_tensor_ptr.data();
        
        for (size_t i = 0; i < m_output_sample_counts.size(); ++i) {
            size_t model_out = m_anira_processor->output_sizes[i];
            m_output_sample_counts[i] = (model_out > 0 && model_out < m_host_buffer_size) ? model_out : m_host_buffer_size;
        }

        m_anira_processor->process(input_data, m_input_sample_counts.data(), 
                                   output_data, m_output_sample_counts.data());
    }
    
    size_t num_output_channels = m_sig_outlets.size();
    for (size_t channel = 0; channel < num_output_channels && channel < output.channel_count(); ++channel) {
        size_t tensor_idx = m_sig_outlets[channel].tensor_index;
        size_t samples_retrieved = m_output_sample_counts[tensor_idx];

        for (size_t sample = 0; sample < sample_count; ++sample) {
            if (anira_initialized) {
                m_dry_wet_mixer.push_dry_sample(m_dry_audio_data[channel < num_channels ? channel : 0][sample], channel);
                
                float wet_sample = 0.0f;
                if (samples_retrieved > 0) {
                    size_t read_idx = (sample < samples_retrieved) ? sample : (samples_retrieved - 1);
                    wet_sample = m_wet_audio_data[channel][read_idx];
                } else {
                    wet_sample = m_last_valid_output[channel];
                }
                
                float value = m_dry_wet_mixer.mix_wet_sample(wet_sample, channel);
                output.samples(channel)[sample] = static_cast<double>(value);
            } else {
                if (channel < num_channels) {
                    output.samples(channel)[sample] = input.samples(channel)[sample];
                } else {
                    output.samples(channel)[sample] = 0.0;
                }
            }
        }
        
        if (samples_retrieved > 0) {
            m_last_valid_output[channel] = m_wet_audio_data[channel][samples_retrieved - 1];
        }
    }
    
    if (m_anira_model_load_confirmed.load(std::memory_order_acquire) == false) {
        m_anira_model_load_confirmed.store(true, std::memory_order_release);
    }
}


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
            std::string type = "";
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
            std::string type = "";
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
    m_host_buffer_size = host_buffer_size;
    float latency = 0.f;
    
    if(m_anira_processor) {
        m_anira_processor->prepare(host_buffer_size, host_sample_rate);
        latency = static_cast<float>(m_anira_processor->get_latency_samples());
        
        prepare_audio_buffers();
        prepare_latency_outlet(latency);
    }

    m_dry_wet_mixer.prepare(host_sample_rate, host_buffer_size, m_sig_outlets.size(), latency);
    m_anira_ready_to_process.store(true, std::memory_order_release);
    
    c74::max::post("anira~: Prepared with %zu signal inlets and %zu signal outlets", 
                   m_sig_inlets.size(), m_sig_outlets.size());
}

void AniraTilde::prepare_audio_buffers() {
    m_dry_audio_data.clear();
    m_wet_audio_data.clear();
    m_input_channel_ptr.clear();
    m_input_tensor_ptr.clear();
    m_output_channel_ptr.clear();
    m_output_tensor_ptr.clear();
    m_input_sample_counts.clear();
    m_output_sample_counts.clear();
    
    // Input
    size_t input_channel_offset = 0;
    for (size_t tensor_idx = 0; tensor_idx < m_anira_processor->inSigCh.size(); ++tensor_idx) {
        const size_t num_channels = m_anira_processor->inSigCh[tensor_idx];
        
        std::vector<float*> channel_pointers;
        for (size_t channel = 0; channel < num_channels; ++channel) {
            m_dry_audio_data.emplace_back(m_host_buffer_size, 0.0f);
            channel_pointers.push_back(m_dry_audio_data[input_channel_offset + channel].data());
        }
        m_input_channel_ptr.push_back(channel_pointers);
        input_channel_offset += num_channels;
        
        // c74::max::post("anira~: Input Tensor %zu: %zu channel × %zu sample (model expects %zu)", 
        //               tensor_idx, num_channels, m_host_buffer_size, m_anira_processor->input_sizes[tensor_idx]);
    }
    
    for (size_t tensor_idx = 0; tensor_idx < m_input_channel_ptr.size(); ++tensor_idx) {
        m_input_tensor_ptr.push_back(m_input_channel_ptr[tensor_idx].data());
        m_input_sample_counts.push_back(m_host_buffer_size);
    }
    
    // Output
    size_t output_channel_offset = 0;
    for (size_t tensor_idx = 0; tensor_idx < m_anira_processor->outSigCh.size(); ++tensor_idx) {
        const size_t num_channels = m_anira_processor->outSigCh[tensor_idx];
        
        std::vector<float*> channel_pointers;
        for (size_t channel = 0; channel < num_channels; ++channel) {
            m_wet_audio_data.emplace_back(m_host_buffer_size, 0.0f);
            channel_pointers.push_back(m_wet_audio_data[output_channel_offset + channel].data());
        }
        m_output_channel_ptr.push_back(channel_pointers);
        output_channel_offset += num_channels;
        
        // c74::max::post("anira~: Output Tensor %zu: %zu channel × %zu sample (model expects %zu)", 
        //               tensor_idx, num_channels, m_host_buffer_size, m_anira_processor->output_sizes[tensor_idx]);
    }
    
    for (size_t tensor_idx = 0; tensor_idx < m_output_channel_ptr.size(); ++tensor_idx) {
        m_output_tensor_ptr.push_back(m_output_channel_ptr[tensor_idx].data());
        
        const size_t model_out = m_anira_processor->output_sizes[tensor_idx];
        const size_t request_size = (model_out > 0 && model_out < m_host_buffer_size) ? model_out : m_host_buffer_size;
        m_output_sample_counts.push_back(request_size);
    }

    m_last_valid_output.assign(m_wet_audio_data.size(), 0.0f);
}

void AniraTilde::prepare_latency_outlet(float latency) {
    for(const auto& outlet : m_msg_outlets) {
        if(outlet.type == MaxType::LATENCY) {
            outlet.outlet->send(static_cast<int>(latency));
            break;
        }
    }
}

void AniraTilde::parse_input_messages(int inlet_num, const std::vector<float>& args) {
    // c74::max::post("anira~: Received data on inlet %d", inlet_num);

    const size_t num_sig_inputs = m_sig_inlets.size();
    const size_t msg_index = static_cast<size_t>(inlet_num) - num_sig_inputs;

    const size_t expected = m_msg_inlets[msg_index].num_channels;
    if (args.size() != expected) {
        c74::max::error("anira~: Incorrect number of elements for inlet %zu. Expected %zu but got %zu.",
                        msg_index + 1, expected, args.size());
        return;
    }

    for (size_t i = 0; i < args.size(); ++i) {
        m_anira_processor->set_input(args[i], m_msg_inlets[msg_index].tensor_index, i);
    }
}

MIN_EXTERNAL_CUSTOM(AniraTilde, anira~);
