#pragma once
#include <map>
#include <string>
#include <vector>

#include "ComputeEngine/matrix.hpp"
#include "util/json.hpp"

namespace GPUCompute {
    class Algorithm : public JSON::JsonConfig {
    public:
        int getNumberOfSweeps();
        virtual void reset();
        // virtual void print() = 0;
        void setSweepParameters(int index);
        // virtual void saveModel(std::string name) = 0;
        // virtual void loadModel(std::string name) = 0;
        // virtual void loadModel(const std::vector<Matrix> &P) = 0;
        std::string parameterString();
        void printJsonParameters();
        void printHeader();

        explicit Algorithm(json &params);
        json parameters;
    protected:
        bool isInitialized = false;
    };
}

