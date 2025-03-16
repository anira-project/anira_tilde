#ifndef CUSTOM_CNN_PRE_POST_PROCESSOR
#define CUSTOM_CNN_PRE_POST_PROCESSOR

#include <anira/anira.h>

class CNNPrePostProcessor : public anira::PrePostProcessor
{
public:
    CNNPrePostProcessor(anira::InferenceConfig cnn_config) : m_inference_config(cnn_config)
    {
        m_inference_config = cnn_config;
    }

    virtual void pre_process(anira::RingBuffer& input, anira::AudioBufferF& output, [[maybe_unused]] anira::InferenceBackend current_inference_backend) override {
        pop_samples_from_buffer(input, output, m_inference_config.m_output_sizes[m_inference_config.m_index_audio_data[anira::IndexAudioData::Output]], m_inference_config.m_input_sizes[m_inference_config.m_index_audio_data[anira::IndexAudioData::Input]]-m_inference_config.m_output_sizes[m_inference_config.m_index_audio_data[anira::IndexAudioData::Output]]);
    }

    anira::InferenceConfig m_inference_config;
};


#endif //CUSTOM_CNN_PRE_POST_PROCESSOR
