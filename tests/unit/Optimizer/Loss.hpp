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
    auto l = Loss::leastSquaresLoss(CE, inputs[0], inputs[1], 1);
    auto outputs = CE.run(inputs, {m, n}, {l});

    Numeric_t e = 2.0;
    EXPECT_TRUE(_::isClose(outputs[0](0, 0), e));
}

int inThresh(const Numeric_t x, const Numeric_t t, const Numeric_t thresh) {
    return std::abs(x - t) < thresh ? 1 : 0;
}

Numeric_t threshlog(const Numeric_t x, const Numeric_t thresh = 1e-4) {
    return inThresh(x, 0.0, thresh) == 1 ? 0 : log(x);
}

Numeric_t crossEntropy(const MatrixRef P, const MatrixRef Y) {
    const int numsamples = P.cols();
    auto expr = [](Numeric_t p, Numeric_t y) -> Numeric_t {
        return (y*threshlog(p) + (1.0 - y)*threshlog(1.0 - p));
    };
    const Numeric_t los = P.binaryExpr(Y, expr).sum();
    return -1.0 * (los / static_cast<Numeric_t>(numsamples));
}

// we should be able to compute the cross entropy loss between two matrices
TEST(Loss, crossEntopyLoss) {
    ComputeEngine CE;

    Matrix m(3, 3);
    m << 1, .5, 0,
         1, 0, 0,
         0, 0, 1;

    Matrix n(3, 3);
    n << 1, 0, 0,
         1, 0, 0,
         0, 0, 1;

    auto inputs = CE.InputVariables(2);
    auto l = Loss::crossEntropyLoss(CE, inputs[0], inputs[1], 1);
    auto outputs = CE.run(inputs, {m, n}, {l});

    EXPECT_TRUE(_::isClose(outputs[0](0, 0) / static_cast<Numeric_t>(3.0), crossEntropy(m, n)));
}
