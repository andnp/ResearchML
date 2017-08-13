#pragma once
#include "ComputeEngine/ComputeEngine.hpp"

namespace GPUCompute {
    tensorflow::Output leastSquaresLoss(ComputeEngine &CE, Input Yhat, Input Y);
}
