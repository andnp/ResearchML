#include "loss.hpp"
#include "ComputeEngine/ComputeEngine.hpp"

namespace GPUCompute {
namespace Loss {
    TFNode leastSquaresLoss(ComputeEngine &CE, Input Yhat, Input Y) {
        auto d = CE.SquaredDifference(Y, Yhat);
        auto s = CE.Sum(CE.Sum(d, 1), 0);
        return s;
    }

    TFNode logisticError(ComputeEngine &CE, TFNode X, TFNode Y, TFNode W) {
        auto Z = CE.MatMul(W, X);
        auto S = CE.Sigmoid(Z);
        auto E = CE.Sub(S, Y);
        return E;
    }
}}
