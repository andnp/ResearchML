#include "gtest/gtest.h"
#include "gpucompute.hpp"

TEST(ComputeEngine, matrix2tensor2matrix) {
    using namespace GPUCompute;
    ComputeEngine CE;

    Matrix m(2, 2);
    m << 1,2,3,4;

    auto t = CE.getTensorFromMatrix(m);
    auto n = CE.getMatrixFromTensor(t);

    zipMatrices(m, n, [](auto v1, auto v2, auto i, auto j) {
        EXPECT_EQ(v1, v2);
        return 0;
    });
}

TEST(ComputeEngine, add) {
    using namespace GPUCompute;
    ComputeEngine CE;

    Matrix m(2, 2);
    m << 1, 2, 3, 4;

    auto t = CE.getTensorFromMatrix(m);

    auto a = CE.InputVariable();
    auto b = CE.InputVariable();
    auto c = CE.Add(a, b);

    auto outputs = CE.run({{a, t}, {b, t}}, {c});
    auto o = CE.getMatrixFromTensor(outputs[0]);

    Matrix e(2, 2);
    e << 2, 4, 6, 8;


    zipMatrices(o, e, [](auto v1, auto v2, auto i, auto j) {
        EXPECT_EQ(v1, v2);
        return 0;
    });
}

TEST(ComputeEngine, addDifferent) {
    using namespace GPUCompute;
    ComputeEngine CE;

    Matrix m(2, 2);
    m << 1, 2, 3, 4;

    Matrix n(2, 2);
    n << 4, 3, 2, 1;

    auto inputs = CE.InputVariables(2);
    auto c = CE.Add(inputs[0], inputs[1]);
    auto outputs = CE.run(inputs, {m, n}, {c});

    Matrix e = m + n;

    auto o = outputs[0];

    zipMatrices(o, e, [](auto v1, auto v2, auto i, auto j) {
        EXPECT_EQ(v1, v2);
        return 0;
    });
}

/** Input Eigen::Matrix values as inputs to the tensorflow graph
*** Outputs Eigen::Matrix values as outputs to the tensorflow graph
*** Makes tensorflow graphs way easier to work with, don't have to think about matrix -> tensor -> matrix conversions anymore
**/
TEST(ComputeEngine, implicitRun) {
    using namespace GPUCompute;
    ComputeEngine CE;

    Matrix m(2, 2);
    m << 1, 2, 3, 4;


    auto inputs = CE.InputVariables(2);
    auto c = CE.Add(inputs[0], inputs[1]);
    auto outputs = CE.run(inputs, {m, m}, {c});

    Matrix e(2, 2);
    e << 2, 4, 6, 8;

    auto o = outputs[0];

    zipMatrices(o, e, [](auto v1, auto v2, auto i, auto j) {
        EXPECT_EQ(v1, v2);
        return 0;
    });
}

