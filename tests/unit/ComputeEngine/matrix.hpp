#include "gtest/gtest.h"
#include "gpucompute.hpp"

TEST(matrix, iterateMatrix) {
    using namespace GPUCompute;
    Matrix m(4, 2);
    m <<    1, 2,
            3, 4,
            5, 6,
            7, 8;

    iterateMatrix(m, [&m](auto v, auto i, auto j) {
        EXPECT_EQ(m(i, j), v);
        return 0;
    });
}

TEST(matrix, zipMatrices) {
    using namespace GPUCompute;
    Matrix m(2, 2);
    Matrix n(2, 2);

    m <<    1, 2,
            3, 4;

    n <<    5, 6,
            7, 8;

    zipMatrices(m, n, [&m, &n](auto v1, auto v2, auto i, auto j) {
        EXPECT_EQ(m(i, j), v1);
        EXPECT_EQ(n(i, j), v2);
        return 0;
    });
}

TEST(matrix, fillWithRandom) {
    using namespace GPUCompute;
    Matrix m = Matrix::Zero(10, 10);
    Matrix n = Matrix::Zero(10, 10);

    fillWithRandom(m);
    fillWithRandom(n);

    zipMatrices(m, n, [](auto v1, auto v2, auto i, auto j) {
        EXPECT_NE(v1, v2);
        return 0;
    });
}
