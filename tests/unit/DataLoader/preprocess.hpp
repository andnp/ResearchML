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
    oneHot(m, n);

    EXPECT_TRUE(areMatricesEqual(n, e));
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
    oneHot(m, n);

    EXPECT_TRUE(areMatricesEqual(n, e));
}
