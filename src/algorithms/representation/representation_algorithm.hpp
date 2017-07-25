#pragma once
#include "ComputeEngine/matrix.hpp"

namespace GPUCompute {
    class RepresentationAlgorithm {
    public:
        virtual Matrix getRepresentation(const Matrix &X) = 0;
    };
}
