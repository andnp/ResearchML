#include "gtest/gtest.h"
#include "gpucompute.hpp"
using namespace GPUCompute;

TEST(matrix, iterateMatrix) {
    Matrix m(4, 2);
    m <<    1, 2,
            3, 4,
            5, 6,
            7, 8;

    MatrixUtil::iterateMatrix(m, [&m](auto v, auto i, auto j) {
        EXPECT_EQ(m(i, j), v);
        return 0;
    });
}

TEST(matrix, zipMatrices) {
    Matrix m(2, 2);
    Matrix n(2, 2);

    m <<    1, 2,
            3, 4;

    n <<    5, 6,
            7, 8;

    MatrixUtil::zipMatrices(m, n, [&m, &n](auto v1, auto v2, auto i, auto j) {
        EXPECT_EQ(m(i, j), v1);
        EXPECT_EQ(n(i, j), v2);
        return 0;
    });
}

TEST(matrix, fillWithRandom) {
    Matrix m = Matrix::Zero(10, 10);
    Matrix n = Matrix::Zero(10, 10);

    MatrixUtil::fillWithRandom(m);
    MatrixUtil::fillWithRandom(n);

    MatrixUtil::zipMatrices(m, n, [](auto v1, auto v2, auto i, auto j) {
        EXPECT_NE(v1, v2);
        return 0;
    });
}

TEST(matrix, stackCols) {
    Matrix m(2, 2);
    Matrix n(2, 2);
    Matrix e(2, 4);
    m << 1, 2,
         5, 6;
    n << 3, 4,
         7, 8;

    e << 1, 2, 3, 4,
         5, 6, 7, 8;

    auto o = MatrixUtil::stackCols({m, n});
    std::cout << o << std::endl;
    EXPECT_TRUE(MatrixUtil::areMatricesEqual(e, o));
}
