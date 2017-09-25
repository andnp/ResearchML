#pragma once
#include <string>
#include "util/json.hpp"

namespace GPUCompute {
class Experiment : public JSON::JsonConfig {
   public:
    int index = -1;
    inline Experiment(json &j) { setConfig(j); }
};

namespace ExperimentParser {
    json getParameters(json &e, int i);
    int numParameters(json exp);
}}
