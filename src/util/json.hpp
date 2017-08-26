#pragma once
#include "json/src/json.hpp"

// Very thin wrapper around json lib.
// Makes it easy to change libs if necessary in the future

namespace GPUCompute {
    using nlohmann::json;

namespace JSON {
    std::string getJsonString(const json &js);
    void extendJson(json &j1, const json &j2);

    class JsonConfig {
        virtual json getDefault() = 0;
    public:
        json config;
        void setConfig(json &j);
    };
}}
