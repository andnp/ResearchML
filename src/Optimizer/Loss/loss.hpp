#pragma once
#include "ComputeEngine/ComputeEngine.hpp"

namespace GPUCompute {
namespace Loss {
    TFNode leastSquaresLoss(ComputeEngine &CE, Input Yhat, Input Y, int samples);
    TFNode crossEntropyLoss(ComputeEngine &CE, TFNode Yhat, TFNode Y, int samples);
    TFNode leastSquaresLossGradient(ComputeEngine &CE, Input Yhat, Input Y, int samples);
    TFNode gradientCrossEntropy(ComputeEngine &CE, TFNode Yhat, TFNode Y, int samples);

    TFNode sigmoidGradient(ComputeEngine &CE, TFNode X);
}}
