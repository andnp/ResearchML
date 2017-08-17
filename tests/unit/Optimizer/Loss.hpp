#include "gtest/gtest.h"
#include "gpucompute.hpp"
using namespace GPUCompute;

// we should be able to compute the least squares loss between two matrices
TEST(Loss, leastSquaresLoss) {
    ComputeEngine CE;

    Matrix m(3, 3);
    m << 0, 1, 0,
         1, 0, 0,
         0, 0, 1;

    Matrix n(3, 3);
    n << 1, 0, 0,
         1, 0, 0,
         0, 0, 1;

    auto inputs = CE.InputVariables(2);
    auto l = leastSquaresLoss(CE, inputs[0], inputs[1]);
    auto outputs = CE.run(inputs, {m, n}, {l});

    EXPECT_TRUE(_::isClose(outputs[0](0, 0), 2.0));
}
