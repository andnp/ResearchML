#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include "gpucompute.hpp"

using namespace GPUCompute;

void toy() {
    ComputeEngine CE;

    Scope root = CE.getScope();
    ClientSession *session = CE.getSession();
    // Matrix A = [3 2; -1 0]
    auto A = CE.InputVariable();
    // Vector b = [3 5]
    Matrix I(1, 2);
    I << 3.0, 5.0;
    auto b = CE.Const(I);

    auto var = CE.Var({2, 2}, CE.Const({{0.0, 0.0}, {0.0, 0.0}}));
    auto Accum = CE.AssignAdd(var, Const(root, {{1.0, 1.0}, {1.0, 1.0}}));

    auto var2 = CE.Var({2, 2}, CE.Const({{1.0, 0.0}, {0.0, 1.0}}));
    auto Increment = CE.AssignAdd(var2, var2);

    auto v = CE.MatMul(A, b, MatMul::TransposeB(true));
    auto s = CE.Sigmoid(v);
    auto o = CE.Div(s, 2.0);
    auto temp = CE.Add(A, CE.Add(Accum, Increment));

    Matrix Amat(2, 2);
    Amat << 3, 2, -1, 0;

    Tensor T = CE.getTensorFromMatrix(Amat);

    std::cout << Amat << std::endl;
    std::cout << CE.getMatrixFromTensor(T) << std::endl;

    CE.run({{A, T}}, {o, temp});
    CE.run({{A, T}}, {o, temp});
    CE.run({{A, T}}, {o, temp});
    CE.run({{A, T}}, {o, temp});
    CE.run({{A, T}}, {o, temp});
    CE.run({{A, T}}, {o, temp});
    CE.run({{A, T}}, {o, temp});
    CE.run({{A, T}}, {o, temp});
    auto outputs = CE.run({{A, T}}, {o, temp});

    // Expect outputs[0] == [19; -3]
    Matrix P = CE.getMatrixFromTensor(outputs[0]);
    std::cout << P << std::endl;

    Matrix Temp = CE.getMatrixFromTensor(outputs[1]);
    std::cout << Temp << std::endl;
}

template<typename T>
T sum(const std::vector<T> &v) {
    T total = 0;
    for (int i = 0; i < v.size(); ++i) total += v[i];
    return total;
}

template<typename T>
void add(std::vector<T> &v, const T num) {
    for (int i = 0; i < v.size(); ++i) v[i] = v[i] + num;
}

std::vector<int> colMaxes(const Matrix &I) {
    std::vector<int> maxes = {};
    const int cols = I.cols();
    for (int i = 0; i < cols; ++i) maxes.push_back(I.col(i).maxCoeff());
    return maxes;
}

void oneHot(const Matrix &I, Matrix &O) {
    const int rows = I.rows();
    const int cols = I.cols();
    auto maxes = colMaxes(I);
    add(maxes, 1);
    const int numNewCols = sum(maxes);
    int offset = 0;
    O = Matrix::Zero(rows, numNewCols);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            O(i, offset + I(i, j)) = 1;
        }
        offset += maxes[j];
    }
}

template<class Func_t>
void splitMinibatch(ComputeEngine &CE, Input X, Input Y, int samples, int batch_size, Func_t f) {
    const int num_splits = ceil(samples / batch_size);
    auto Splits = CE.Concat({CE.Fill({num_splits - 1}, batch_size), {-1}}, 0);
    auto X_batches = CE.SplitV(X, Splits, 1, num_splits);
    auto Y_batches = CE.SplitV(Y, Splits, 1, num_splits);
    int accum_samples = 0;
    for (int i = 0; i < num_splits; ++i) {
        // assume each batch as batch_size number of samples
        int batch_samples = batch_size;
        // on the last batch we may have fewer samples if not evenly divisible. Account for that here
        if (i == num_splits - 1) batch_samples = samples - accum_samples;
        accum_samples += batch_samples;

        f(CE, X_batches[i], Y_batches[i], batch_samples);
    }
}

tensorflow::Output leastSquaresLoss(ComputeEngine &CE, Input Yhat, Input Y) {
    auto d = CE.Sub(Y, Yhat);
    auto m = CE.Multiply(d, d);
    auto s = CE.Sum(CE.Sum(m, 1), 0);
    return CE.Sqrt(s);
}

void LR() {
    const std::string data_path = "~/Projects/research/ml_data/cifar10.csv";
    Matrix CIFAR;
    readMatrix(data_path, CIFAR);

    Matrix X = CIFAR.block(0, 1, 60000, 1024);
    Matrix Yindex = CIFAR.block(0, 0, 60000, 1);
    Matrix Y;
    oneHot(Yindex, Y);

    const int samples = X.rows();
    const int features = X.cols();
    const int classes = Y.cols();

    X.transposeInPlace();
    Y.transposeInPlace();

    Matrix w(classes, features);
    fillWithRandom(w, 0, 1);

    ComputeEngine CE;

    // auto XT = CE.getTensorFromMatrix(X);
    // auto YT = CE.getTensorFromMatrix(Y);
    auto WT = CE.getTensorFromMatrix(w);

    auto W = CE.Var(w);

    // auto X_ = CE.InputVariable();
    // auto Y_ = CE.InputVariable();

    auto X_ = CE.Var(X);
    auto Y_ = CE.Var(Y);
    // auto NW = CE.Copy(W);

    splitMinibatch(CE, X_, Y_, 60000, 100, [&W](auto &CE, auto X, auto Y, int batch_samples) {
        Numeric_t scaling = static_cast<Numeric_t>(batch_samples);

        auto Z = CE.MatMul(W, X);
        auto S = CE.Sigmoid(Z);
        auto E = CE.Sub(S, Y);
        auto G = CE.Div(CE.MatMul(E, X, MatMul::TransposeB(true)), scaling);
        auto GR = CE.Add(CE.Multiply(W, 2.0 * 0), G);

        W = CE.AssignSub(W, CE.Multiply(GR, 0.01));
    });


    // auto W_ = CE.Assign(W, NW);

    auto Yhat = CE.Sigmoid(CE.MatMul(W, X_));
    auto NE = leastSquaresLoss(CE, Yhat, Y_);

    const int steps = 10000;
    for (int s = 0; s < steps / 100; ++s) {
        auto outs = CE.run({}, {}, {NE});
        std::cout << outs[0] << std::endl;
    }
}

int main() {
    LR();

    return 0;
}
