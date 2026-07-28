#include "anira_tilde/resampling/ResamplerConfigParser.h"

#include <cstddef>
#include <fstream>
#include <istream>
#include <map>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "anira_tilde/resampling/Resampler.h"

namespace anira_tilde {

namespace {

ResamplerQuality quality_from_string(const std::string& q, ResamplerQuality fallback) {
    if (q == "sinc_best") { return ResamplerQuality::SincBest; }
    if (q == "sinc_medium") { return ResamplerQuality::SincMedium; }
    if (q == "sinc_fastest") { return ResamplerQuality::SincFastest; }
    if (q == "linear") { return ResamplerQuality::Linear; }
    if (q == "hold") { return ResamplerQuality::Hold; }
    return fallback;
}

void parse_quality_map(const nlohmann::json& rs,
                       const char* key,
                       ResamplerQuality fallback,
                       std::map<size_t, ResamplerQuality>& out) {
    if (!rs.contains(key)) { return; }
    for (const auto& [tensor, value] : rs.at(key).items()) {
        out[static_cast<size_t>(std::stoul(tensor))] =
            quality_from_string(value.get<std::string>(), fallback);
    }
}

}  // namespace

ResamplerConfig parse_resampler_config(std::istream& stream) {
    ResamplerConfig result;
    try {
        nlohmann::json config;
        stream >> config;

        if (!config.contains("resampler_config")) { return result; }
        const auto& rs = config.at("resampler_config");
        if (rs.contains("model_sample_rate")) {
            result.m_model_sample_rate = rs.at("model_sample_rate").get<double>();
        }
        if (rs.contains("quality")) {
            result.m_quality =
                quality_from_string(rs.at("quality").get<std::string>(), result.m_quality);
        }
        parse_quality_map(rs, "input_quality", result.m_quality, result.m_input_quality);
        parse_quality_map(rs, "output_quality", result.m_quality, result.m_output_quality);
    } catch (...) {  // NOLINT(bugprone-empty-catch) malformed JSON or wrong types: keep defaults
    }
    return result;
}

ResamplerConfig parse_resampler_config(const std::string& json_path) {
    if (json_path.empty()) { return {}; }
    std::ifstream file(json_path);
    if (!file.is_open()) { return {}; }
    return parse_resampler_config(file);
}

}  // namespace anira_tilde
