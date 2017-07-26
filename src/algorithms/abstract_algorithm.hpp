#pragma once
#include <map>
#include <string>
#include <vector>

#include "ComputeEngine/matrix.hpp"
#include "util/json.hpp"

namespace GPUCompute {
    class Algorithm {
    public:
        virtual void optimize(int steps) = 0;
        int getNumberOfSweeps();
        virtual void reset() = 0;
        virtual void print() = 0;
        virtual void setSweepParameters(int index);
        virtual void saveModel(std::string name);
        virtual void loadModel(std::string name);
        virtual void loadModel(const std::vector<Matrix> &P);
        virtual std::map<std::string, float> getDefaults();
        std::string parameterString();
        void printJsonParameters();
        void printHeader();
        void loadParameters(json j);

        json parameters;

        std::map<const std::string, std::vector<float>> sweeps;

        std::string task = "classification";

    protected:
        int isInitialized = 0;

        virtual void setup() = 0;
    };
}

