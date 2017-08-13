#include "loss.hpp"
#include "ComputeEngine/ComputeEngine.hpp"

namespace GPUCompute {
    tensorflow::Output leastSquaresLoss(ComputeEngine &CE, Input Yhat, Input Y) {
        auto d = CE.Sub(Y, Yhat);
        auto m = CE.Multiply(d, d);
        auto s = CE.Sum(CE.Sum(m, 1), 0);
        return s;
    }
}
