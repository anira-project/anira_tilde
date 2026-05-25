#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "anira_tilde/Exports.h"
#include "anira_tilde/mixing/Mixer.h"
#include "anira_tilde/inference/Session.h"

namespace anira_tilde {

/**
 * @brief Host-agnostic inference engine for the anira~ external.
 *
 * Wraps a Session (anira pipeline + state passing + rate adaptation)
 * together with the dry/wet Mixer and per-tensor audio scratch buffers.
 * Float-only — hosts that work in double (Max/MSP) cast at their boundary.
 *
 * Lifecycle: construct with the JSON config, call prepare() from the
 * host's DSP setup, then call process() per audio block.
 */
class ANIRA_TILDE_API Engine {
public:
    Engine() = default;
    ~Engine() = default;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    /// Load a JSON config and bring the underlying Session online.
    /// Returns true on success. On failure, error_out (if non-null) is
    /// populated with the exception message and the Engine remains
    /// in its empty / not-loaded state.
    bool load_config(const std::string& json_config_path, std::string* error_out = nullptr);
    bool config_loaded() const noexcept { return m_session != nullptr; }

    // --- Tensor layout (used by hosts to set up inlets/outlets) ----------
    const std::vector<size_t>&              sig_input_channels()  const;
    const std::vector<size_t>&              sig_output_channels() const;
    const std::vector<std::vector<size_t>>& msg_input_channels()  const;
    const std::vector<std::vector<size_t>>& msg_output_channels() const;
    size_t state_pair_count() const;

    // --- Lifecycle -------------------------------------------------------
    void   prepare(size_t host_buffer_size, double host_sample_rate);
    bool   ready() const noexcept { return m_ready.load(std::memory_order_acquire); }
    size_t latency_samples() const;

    // --- Block processing (float-native) --------------------------------
    void process(const float* const* input,  size_t num_input_channels,
                 float* const*       output, size_t num_output_channels,
                 size_t sample_count);

    // --- Non-streamable (message) tensor accessors -----------------------
    void  set_message_input (size_t tensor_index, const std::vector<float>& values);
    float get_message_output(size_t tensor_index, size_t channel);

    // --- Dry/wet mixer ---------------------------------------------------
    void set_dry_wet_mix(float mix01);
    bool mixing_disabled() const noexcept { return m_mixing_disabled; }

private:
    void prepare_audio_buffers();
    void process_anira();
    void write_mixed_output(const float* const* input,  size_t num_input_channels,
                            float* const*       output, size_t num_output_channels,
                            size_t sample_count, bool ready,
                            const TensorLayout& layout);

    std::unique_ptr<Session> m_session;
    Mixer                    m_mixer;

    std::vector<anira::Buffer<float>>  m_input_buffers;
    std::vector<anira::Buffer<float>>  m_output_buffers;

    std::vector<const float* const*>   m_input_tensor_ptrs;
    std::vector<float* const*>         m_output_tensor_ptrs;

    std::vector<size_t>  m_input_sample_counts;
    std::vector<size_t>  m_output_sample_counts;

    size_t            m_host_buffer_size = 0;
    bool              m_mixing_disabled  = false;
    std::atomic<bool> m_ready{false};
};

} // namespace anira_tilde
