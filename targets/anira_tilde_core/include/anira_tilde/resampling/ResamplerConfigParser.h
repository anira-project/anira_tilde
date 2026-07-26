#pragma once

#include <istream>
#include <map>
#include <string>

#include "anira_tilde/Exports.h"
#include "anira_tilde/resampling/Resampler.h"

namespace anira_tilde {

/**
 * @brief Parsed "resampler_config" section of a JSON model config.
 *
 * When model_sample_rate is > 0 and differs from the host rate at prepare
 * time, the Session converts every streamable tensor host→model on the way in
 * and model→host on the way out. 0 (or an absent section) disables resampling.
 *
 * JSON:
 *   "resampler_config": {
 *       "model_sample_rate": 44100,
 *       "quality": "sinc_fastest",       // sinc_best | sinc_medium | sinc_fastest | linear | hold
 *       "input_quality":  { "1": "hold" },  // optional per-tensor overrides
 *       "output_quality": { "0": "hold" }
 *   }
 *
 * Per-tensor overrides make encoder/decoder models work: the audio tensor gets
 * a real (sinc) resampler while latent tensors use "hold" — a zero-order-hold
 * count adapter that paces the latent stream to the host rate with exact
 * fractional cadence, preserves latent values exactly (no filter ringing) and
 * adds no latency.
 */
struct ResamplerConfig {
    double model_sample_rate = 0.0;
    ResamplerQuality quality = ResamplerQuality::SincFastest;
    std::map<size_t, ResamplerQuality> input_quality;   // tensor index -> override
    std::map<size_t, ResamplerQuality> output_quality;  // tensor index -> override

    ResamplerQuality quality_for_input(size_t tensor) const {
        auto it = input_quality.find(tensor);
        return it == input_quality.end() ? quality : it->second;
    }
    ResamplerQuality quality_for_output(size_t tensor) const {
        auto it = output_quality.find(tensor);
        return it == output_quality.end() ? quality : it->second;
    }
};

ANIRA_TILDE_API ResamplerConfig parse_resampler_config(std::istream& stream);
ANIRA_TILDE_API ResamplerConfig parse_resampler_config(const std::string& json_path);

}  // namespace anira_tilde
