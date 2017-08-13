#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/cc/ops/standard_ops.h>

/**
*   This file sets a few global defaults for matrices.
*   Allows for easily changing the number of bits used for computations amongst other settings
*/

namespace GPUCompute {
    using namespace tensorflow;
    using namespace tensorflow::ops;

    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
    using MatrixRef = Eigen::Ref<const Matrix>;
    extern DataType Tensor_t;
    typedef double Numeric_t;

    std::vector<int> getDimensions(std::string file);
    void readMatrix(std::string file, Matrix &X);
    void fillWithRandom(Matrix &M);
    void fillWithRandom(Matrix &M, float mean, float sigma);

    template<class Func_t>
    Matrix iterateMatrix(Matrix &M, Func_t f) {
        const int cols = M.cols();
        const int rows = M.rows();
        Matrix o(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                o(i, j) = f(M(i, j), i, j);
            }
        }
        return o;
    }

    template<class Func_t>
    Matrix zipMatrices(Matrix &M, Matrix &N, Func_t f) {
        const int cols = M.cols();
        const int rows = M.rows();
        Matrix o(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                o(i, j) = f(M(i, j), N(i, j), i, j);
            }
        }
        return o;
    }

    std::vector<int> colMaxes(const Matrix &I);
    bool areMatricesEqual(Matrix a, Matrix b);
}
