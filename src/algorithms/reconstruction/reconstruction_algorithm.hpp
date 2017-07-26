#pragma once
#include "ComputeEngine/matrix.hpp"

namespace GPUCompute {
    class ReconstructionAlgorithm {
    public:
        virtual Matrix getReconstruction() = 0;
    };
}
