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
    using MutMatrixRef = Eigen::Ref<Matrix>;
    using TFNode = tensorflow::Output;
    extern DataType Tensor_t;
    typedef double Numeric_t;

namespace MatrixUtil {
    std::vector<int> getDimensions(std::string file);
    void readMatrix(std::string file, Matrix &X);
    void fillWithRandom(Matrix &M);
    void fillWithRandom(Matrix &M, Numeric_t mean, Numeric_t sigma);

    template<class Func_t>
    Matrix iterateMatrix(MatrixRef M, Func_t f) {
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
    Matrix zipMatrices(MatrixRef M, MatrixRef N, Func_t f) {
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

    inline Matrix stackCols(std::vector<Matrix> v) {
        int total_cols = 0;
        for (int i = 0; i < v.size(); ++i) total_cols += v[i].cols();

        int seen = 0;
        Matrix out(v[0].rows(), total_cols);
        for (int i = 0; i < v.size(); ++i) {
            out.block(0, seen, v[i].rows(), v[i].cols()) = v[i];
            seen += v[i].cols();
        }
        return out;
    }

    std::vector<int> colMaxes(const MatrixRef I);
    bool areMatricesEqual(MatrixRef a, MatrixRef b);
    Matrix getRandomMatrix(int rows, int cols, Numeric_t mean, Numeric_t sigma);
}}
