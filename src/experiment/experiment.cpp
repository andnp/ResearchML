#include "experiment.hpp"
#include "util/json.hpp"

int numParameters(json exp) {
    int num = 1;
    if (exp["sweeps"].is_null()) {
        return 0;
    }
    json sweeps = exp["sweeps"];
    for (const auto& j : json::iterator_wrapper(sweeps)) {
        std::vector<float> sweep = sweeps[j.key()];
        num *= sweep.size();
    }
    return num;
}

Experiment parseExperiment(json params) {
    json experiment = JSON::readFile(params["experiment_file"]);

    int index = -1;
    if (!params["index"].is_null()) {
        index = atoi(params["index"]);
    }

    return parseExperiment(experiment, index);
}

Experiment generateExperiment(json experiment, int index) {
    Experiment e(experiment);


    return e;
}
