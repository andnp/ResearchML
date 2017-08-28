#include <algorithm>
#include <string>
#include <vector>

#include "abstract_algorithm.hpp"
#include "ComputeEngine/matrix.hpp"

#include "util/Logger/Logger.hpp"

namespace GPUCompute {
    int Algorithm::getNumberOfSweeps() {
        int total = 1;
        json sweeps = config["parameters"];
        for (const auto &pair : json::iterator_wrapper(sweeps)) {
            total *= (pair.value()).size();
        }
        return total;
    }

    void Algorithm::setSweepParameters(int index) {
        int accum = 1;
        json sweeps = config["parameters"];
        for (const auto &pair : json::iterator_wrapper(sweeps)) {
            const int num = (pair.value()).size();
            parameters[pair.key()] = (pair.value())[(index / accum) % num];
            accum *= num;
        }
    }

    void Algorithm::printHeader() {
        std::string out = csvHeader(parameters);
        Logger::out() << "samples, features, " << out.substr(0, out.length() - 2) << ", test, train" << std::endl;
    }

    std::string Algorithm::parameterString() {
        std::string out = JSON::getJsonString(parameters);
        return out.substr(0, out.length() - 2);
    }

    void Algorithm::printJsonParameters() {
        std::cout << parameters.dump(4) << std::endl;
    }

    void Algorithm::reset() {
        throw "reset is not yet implemented for this algorithm";
    }

    // void Algorithm::saveModel(std::string name) {
    //     throw std::invalid_argument("saveModel doesn't work for this algorithm yet. Sorry!");
    // }

    // void Algorithm::loadModel(std::string name) {
    //     throw std::invalid_argument("loadModel doesn't work for this algorithm yet. Sorry!");
    // }

    // void Algorithm::loadModel(const std::vector<Matrix> &parameters) {
    //     throw std::invalid_argument("loadModel(std::vector<Matrix>&) is not yet implemented");
    // }

    Algorithm::Algorithm(json &params) {
        setConfig(params);
    }
}

