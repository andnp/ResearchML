#pragma once
#include <string>
#include <vector>
#include "json/src/json.hpp"

// Very thin wrapper around json lib.
// Makes it easy to change libs if necessary in the future

namespace GPUCompute {
    using nlohmann::json;

namespace JSON {
    std::string getJsonString(const json &js);
    void extendJson(json &j1, const json &j2);
    json without(const json j, std::vector<std::string> keys);
    json readFile(std::string path);

    template <class Func_t>
    void forEach(const json j, Func_t f) {
        for (const auto &pair : json::iterator_wrapper(j)) {
            f(pair.key(), pair.value());
        }
    }

    class JsonConfig {
        virtual json getDefault();
    public:
        json config;
        void setConfig(json &j);
    };
}}
