#pragma once
#include "ComputeEngine/ComputeEngine.hpp"

namespace GPUCompute {
namespace Loss {
    TFNode leastSquaresLoss(ComputeEngine &CE, Input Yhat, Input Y);
    TFNode logisticError(ComputeEngine &CE, TFNode X, TFNode Y, TFNode W);
}}
