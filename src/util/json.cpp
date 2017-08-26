#include "json.hpp"

namespace GPUCompute {
namespace JSON {
    std::string getJsonString(const json &js) {
        std::string out = "";
        for (auto &j : json::iterator_wrapper(js)) {
            if (j.value().is_object()) {
                out += getJsonString(j.value());
            } else if (j.value().is_number_integer()) {
                int v = j.value();
                out += std::to_string(v) + ", ";
            } else if (j.value().is_number()) {
                float v = j.value();
                out += std::to_string(v) + ", ";
            } else if (j.value().is_string()) {
                std::string v = j.value();
                out += v + ", ";
            }
        }
        return out;
    }

    void extendJson(json &j1, const json &j2) {
        for (const auto &j : json::iterator_wrapper(j2)) {
            j1[j.key()] = j.value();
        }
    }

    void JsonConfig::setConfig(json &j) {
        config = {};
        extendJson(config, getDefault());
        extendJson(config, j);
    }
}}
