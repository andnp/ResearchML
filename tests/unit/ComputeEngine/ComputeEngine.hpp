#include "gtest/gtest.h"
#include "gpucompute.hpp"
using namespace GPUCompute;

// we should be able to transfer from a cpu matrix to a heterogeneous matrix
TEST(ComputeEngine, matrix2tensor2matrix) {
    ComputeEngine CE;

    Matrix m(2, 2);
    m << 1,2,3,4;

    auto t = CE.getTensorFromMatrix(m);
    auto n = CE.getMatrixFromTensor(t);


    EXPECT_TRUE(MatrixUtil::areMatricesEqual(m, n));
}

// ---
// Add
// ---

// we can add two matrices
TEST(ComputeEngine, add) {
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

    EXPECT_TRUE(MatrixUtil::areMatricesEqual(o, e));
}

// we can add two matrices
TEST(ComputeEngine, addDifferent) {
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

    EXPECT_TRUE(MatrixUtil::areMatricesEqual(o, e));
}

// ---
// Sub
// ---

// we can subtract two matrices
TEST(ComputeEngine, sub) {
    ComputeEngine CE;

    Matrix m(2, 2);
    m << 1, 2, 3, 4;

    Matrix n(2, 2);
    n << 4, 3, 2, 1;

    auto inputs = CE.InputVariables(2);
    auto c = CE.Sub(inputs[0], inputs[1]);
    auto outputs = CE.run(inputs, {m, n}, {c});

    Matrix e = m - n;

    auto o = outputs[0];

    EXPECT_TRUE(MatrixUtil::areMatricesEqual(o, e));
}

// ------
// MatMul
// ------

// we can perform a matrix multiplication operation
TEST(ComputeEngine, MatMul) {
    ComputeEngine CE;

    Matrix m(2, 3);
    m << 1, 2, 3, 4, 5, 6;

    Matrix n (3, 2);
    n << 7, 6, 5, 4, 3, 2;

    auto inputs = CE.InputVariables(2);
    auto c = CE.MatMul(inputs[0], inputs[1]);
    auto outputs = CE.run(inputs, {m, n}, {c});

    Matrix e(2, 2);
    e << 26, 20, 71, 56;

    auto o = outputs[0];
    EXPECT_TRUE(MatrixUtil::areMatricesEqual(o, e));
}

// we can perform a matrix multiplication with the first matrix transposed
TEST(ComputeEngine, MatMul_transposeA) {
    ComputeEngine CE;

    Matrix m(3, 2);
    m << 1, 2, 3, 4, 5, 6;

    Matrix n(3, 2);
    n << 7, 6, 5, 4, 3, 2;

    auto inputs = CE.InputVariables(2);
    auto c = CE.MatMul(inputs[0], inputs[1], MatMul::TransposeA(true));
    auto outputs = CE.run(inputs, {m, n}, {c});

    Matrix e(2, 2);
    e << 37, 28, 52, 40;

    auto o = outputs[0];

    EXPECT_TRUE(MatrixUtil::areMatricesEqual(o, e));
}

// we can perform a matrix multiplication with the second matrix transposed
TEST(ComputeEngine, MatMul_transposeB) {
    ComputeEngine CE;

    Matrix m(2, 3);
    m << 1, 2, 3, 4, 5, 6;

    Matrix n(2, 3);
    n << 7, 6, 5, 4, 3, 2;

    auto inputs = CE.InputVariables(2);
    auto c = CE.MatMul(inputs[0], inputs[1], MatMul::TransposeB(true));
    auto outputs = CE.run(inputs, {m, n}, {c});

    Matrix e(2, 2);
    e << 34, 16, 88, 43;

    auto o = outputs[0];

    EXPECT_TRUE(MatrixUtil::areMatricesEqual(o, e));
}

// ---
// Sum
// ---

// we can get the sum across one axis of a tensor (matrix)
TEST(ComputeEngine, Sum_dim1) {
    ComputeEngine CE;

    Matrix m(2, 2);
    m << 1, 2, 3, 4;

    auto input = CE.InputVariable();
    auto s = CE.Sum(input, 0);
    auto outputs = CE.run({input}, {m}, {s});

    Matrix e(2, 1);
    e << 4, 6;

    auto o = outputs[0];

    EXPECT_TRUE(MatrixUtil::areMatricesEqual(o, e));
}

// we can get the sum across one axis of a tensor (matrix)
TEST(ComputeEngine, Sum_dim2) {
    ComputeEngine CE;

    Matrix m(2, 2);
    m << 1, 2, 3, 4;

    auto input = CE.InputVariable();
    auto s = CE.Sum(input, 1);
    auto outputs = CE.run({input}, {m}, {s});

    Matrix e(2, 1);
    e << 3, 7;

    auto o = outputs[0];

    EXPECT_TRUE(MatrixUtil::areMatricesEqual(o, e));
}

// we can get the sum across both axes of a tensor (matrix)
TEST(ComputeEngine, Sum_bothDims) {
    ComputeEngine CE;

    Matrix m(2, 2);
    m << 1, 2, 3, 4;

    auto input = CE.InputVariable();
    // Note here that the dims need to go from largest inner to smallest outer
    auto s = CE.Sum(CE.Sum(input, 1), 0);
    auto outputs = CE.run({input}, {m}, {s});

    Matrix e(1, 1);
    e << 10;

    auto o = outputs[0];

    EXPECT_TRUE(MatrixUtil::areMatricesEqual(o, e));
}

// --------
// Multiply
// --------

TEST(ComputeEngine, Multiple_broadcast) {
    ComputeEngine CE;

    Matrix m(2, 2);
    m << 1, 2, 3, 4;

    auto input = CE.InputVariable();
    // Note here that the constant value needs to be a double
    auto n = CE.Multiply(input, 2.0);
    auto outputs = CE.run({input}, {m}, {n});

    Matrix e(2, 2);
    e << 2, 4, 6, 8;

    auto o = outputs[0];

    EXPECT_TRUE(MatrixUtil::areMatricesEqual(o, e));
}

TEST(ComputeEngine, Multiple_elementwise) {
    ComputeEngine CE;

    Matrix m(2, 2);
    m << 1, 2, 3, 4;

    auto input = CE.InputVariable();
    // You don't have to use const, just chose to show it off here as an option
    Matrix tmp(2, 2);
    tmp << 4, 3, 2, 1;
    auto n = CE.Const(tmp);
    auto c = CE.Multiply(input, n);
    auto outputs = CE.run({input}, {m}, {c});

    Matrix e(2, 2);
    e << 4, 6, 6, 4;

    auto o = outputs[0];

    EXPECT_TRUE(MatrixUtil::areMatricesEqual(o, e));
}

// ---------
// Transpose
// ---------

TEST(ComputeEngine, Transpose) {
    ComputeEngine CE;

    Matrix m(2, 3);
    m << 1, 2, 3,
         4, 5, 6;

    auto input = CE.InputVariable();
    // Note here that the constant value needs to be a double
    auto n = CE.Transpose(input);
    auto outputs = CE.run({input}, {m}, {n});

    Matrix e(3, 2);
    e << 1, 4,
         2, 5,
         3, 6;

    auto o = outputs[0];

    EXPECT_TRUE(MatrixUtil::areMatricesEqual(o, e));
}

// ---
// Run
// ---

/** Input Eigen::Matrix values as inputs to the tensorflow graph
*** Outputs Eigen::Matrix values as outputs to the tensorflow graph
*** Makes tensorflow graphs way easier to work with, don't have to think about matrix -> tensor -> matrix conversions anymore
**/
TEST(ComputeEngine, implicitRun) {
    ComputeEngine CE;

    Matrix m(2, 2);
    m << 1, 2, 3, 4;


    auto inputs = CE.InputVariables(2);
    auto c = CE.Add(inputs[0], inputs[1]);
    auto outputs = CE.run(inputs, {m, m}, {c});

    Matrix e(2, 2);
    e << 2, 4, 6, 8;

    auto o = outputs[0];

    EXPECT_TRUE(MatrixUtil::areMatricesEqual(o, e));
}

