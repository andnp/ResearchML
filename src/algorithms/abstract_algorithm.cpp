#include <algorithm>
#include <string>
#include <vector>

#include "abstract_algorithm.hpp"
#include "ComputeEngine/matrix.hpp"

#include "util/Logger/Logger.hpp"

namespace GPUCompute {
    int Algorithm::getNumberOfSweeps() {
        int total = 1;
        for (const auto &pair : sweeps) {
            total *= (pair.second).size();
        }
        return total;
    }

    void Algorithm::setSweepParameters(int index) {
        int accum = 1;
        for (const auto &pair : sweeps) {
            const int num = (pair.second).size();
            parameters[pair.first] = (pair.second)[(index / accum) % num];
            accum *= num;
        }
    }

    void Algorithm::printHeader() {
        std::string out = csvHeader(parameters);
        Logger::out() << "samples, features, " << out.substr(0, out.length() - 2) << ", test, train" << std::endl;
    }

    std::string Algorithm::parameterString() {
        std::string out = getJsonString(parameters);
        return out.substr(0, out.length() - 2);
    }

    void Algorithm::loadParameters(json j) {
        extendJson(parameters, j);
    }

    void Algorithm::printJsonParameters() {
        std::cout << parameters.dump(4) << std::endl;
    }

    void Algorithm::saveModel(std::string name) {
        throw std::invalid_argument("saveModel doesn't work for this algorithm yet. Sorry!");
    }

    void Algorithm::loadModel(std::string name) {
        throw std::invalid_argument("loadModel doesn't work for this algorithm yet. Sorry!");
    }

    void Algorithm::loadModel(const std::vector<Matrix> &parameters) {
        throw std::invalid_argument("loadModel(std::vector<Matrix>&) is not yet implemented");
    }
}

