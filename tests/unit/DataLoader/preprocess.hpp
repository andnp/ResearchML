#include "gtest/gtest.h"
#include "gpucompute.hpp"
using namespace GPUCompute;

TEST(preprocess, oneHot_singleColumn) {
    Matrix m(6, 1);
    m << 0, 1, 2, 3 ,4, 5;

    Matrix e(6, 6);
    e <<    1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1;

    Matrix n;
    Preprocess::oneHot(m, n);

    EXPECT_TRUE(MatrixUtil::areMatricesEqual(n, e));
}

TEST(preprocess, oneHot_multipleColumn) {
    Matrix m(3, 2);
    m << 0, 2,
         1, 1,
         2, 0;

    Matrix e(3, 6);
    e <<    1, 0, 0, 0, 0, 1,
            0, 1, 0, 0, 1, 0,
            0, 0, 1, 1, 0, 0;

    Matrix n;
    Preprocess::oneHot(m, n);

    EXPECT_TRUE(MatrixUtil::areMatricesEqual(n, e));
}

TEST(preprocess, Scaler) {
    Matrix m(3, 2);
    m << 1, 3,
         -100, 2,
         -2, -7;

    Matrix e(3, 2);
    e << 0, 1,
         0, 1,
         1, 0;

    json scaler_config = {
        {"min", 0},
        {"max", 1}
    };
    Preprocess::Scaler scaler(scaler_config);
    scaler.inferRange(m);
    scaler.scale(m);

    EXPECT_TRUE(MatrixUtil::areMatricesEqual(m, e));
}

TEST(preprocess, split) {
    Matrix m(10, 1);
    for (int i = 0; i < 10; ++i) m(i, 0) = i;

    std::vector<Matrix> expected = {
        Matrix(3, 1),
        Matrix(3, 1),
        Matrix(4, 1)
    };

    expected[0] << 0, 1, 2;
    expected[1] << 3, 4, 5;
    expected[2] << 6, 7, 8, 9;

    std::vector<Matrix> results = Preprocess::split(m, 3);

    for (int i = 0; i < 3; ++i) EXPECT_TRUE(MatrixUtil::areMatricesEqual(results[i], expected[i]));
}
