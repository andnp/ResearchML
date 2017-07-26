#pragma once
#include <vector>
#include <string>

#include "ComputeEngine/matrix.hpp"
#include "algorithms/abstract_algorithm.hpp"

namespace GPUCompute {
    class UnsupervisedAlgorithm : public Algorithm {
    public:
        int tmp = 0;
    };
}
