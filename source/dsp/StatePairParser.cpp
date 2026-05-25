#include "StatePairParser.h"

#include <fstream>
#include <nlohmann/json.hpp>

std::vector<StatePair> parse_state_pairs(std::istream& stream) {
    std::vector<StatePair> pairs;
    try {
        nlohmann::json config;
        stream >> config;

        if (!config.contains("state_config")) return pairs;
        const auto& state_json = config.at("state_config");
        if (!state_json.contains("state_pairs")) return pairs;

        for (const auto& entry : state_json.at("state_pairs")) {
            if (!entry.contains("output_tensor") || !entry.contains("input_tensor"))
                continue;
            StatePair pair;
            pair.output_tensor = entry.at("output_tensor").get<size_t>();
            pair.input_tensor  = entry.at("input_tensor").get<size_t>();
            pairs.push_back(pair);
        }
    } catch (...) {}
    return pairs;
}

std::vector<StatePair> parse_state_pairs(const std::string& json_path) {
    if (json_path.empty()) return {};
    std::ifstream file(json_path);
    if (!file.is_open()) return {};
    return parse_state_pairs(file);
}
