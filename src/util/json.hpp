#pragma once
#include "json/src/json.hpp"

// Very thin wrapper around json lib.
// Makes it easy to change libs if necessary in the future

namespace GPUCompute {
    using nlohmann::json;

    std::string getJsonString(const json &js);
    void extendJson(json &j1, const json &j2);
}
