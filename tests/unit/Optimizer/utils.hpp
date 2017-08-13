#include "gtest/gtest.h"
#include "gpucompute.hpp"
using namespace GPUCompute;

// we should be able to compute the least squares loss between two matrices
TEST(Optimizer_utils, splitMinibatch) {
    ComputeEngine CE;

    Matrix x(3, 3);
    x << 0, 1, 0,
         1, 0, 0,
         0, 0, 1;

    Matrix y(3, 1);
    y << 1, 0, 0;

    Matrix ex(3, 3);
    ex << 1, 2, 1,
          3, 2, 2,
          3, 3, 4;

    Matrix ey(3, 1);
    ey << 2, 2, 3;

    auto inputs = CE.InputVariables(2);
    auto xo = CE.Var(Matrix::Zeros(3, 3));
    auto yo = CE.Var(Matrix::Zeros(3, 1));

    int samples = 3;
    int batch_size = 1;
    int count = 1;
    splitMinibatch(CE, inputs[0], inputs[1], samples, batch_size, [&count](auto &CE, auto X, auto Y, int batch_samples) {

    });

    auto outputs = CE.run(inputs, {m, n}, {l});

    EXPECT_TRUE(_::isClose(outputs[0](0, 0), 2.0));
}
