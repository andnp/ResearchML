#include "experiment.hpp"
#include "util/json.hpp"

namespace GPUCompute {
int numParameters(json exp) {
    int num = 1;
    json sweeps = exp;
    for (const auto& j : json::iterator_wrapper(sweeps)) {
        num *= sweeps[j.key()].size();
    }
    return num;
}

namespace ExperimentParser {
    json getParameters(json& e, int index) {
        json output = {};
        JSON::extendJson(output, e);

        int accum = 1;
        for (json::iterator it = e.begin(); it != e.end(); ++it) {
            const int num = (it.value()).size();
            output[it.key()] = (it.value())[(index / accum) % num];
            accum *= num;
        }

        return output;
    }
}}
