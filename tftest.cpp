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

TFNode logisticError(ComputeEngine &CE, TFNode X, TFNode Y, TFNode W) {
    auto Z = CE.MatMul(W, X);
    auto S = CE.Sigmoid(Z);
    auto E = CE.Sub(S, Y);
    return E;
}

TFNode gradientGraph(ComputeEngine &CE, TFNode X, TFNode Y, std::vector<TFNode> Parameters, std::vector<int> dims) {
    TFNode W = Parameters[0];
    int samples = dims[0];
    // compute Y - (sig(W * X));
    auto E = logisticError(CE, X, Y, W);
    auto G = CE.Div(CE.MatMul(E, X, MatMul::TransposeB(true)), static_cast<Numeric_t>(samples));
    return CE.Add(CE.Multiply(W, 2.0 * 0.0001), G);
}

void LR() {
    const std::string data_path = "~/Projects/research/ml_data/cifar10.csv";
    Matrix CIFAR;
    MatrixUtil::readMatrix(data_path, CIFAR);

    Matrix X = CIFAR.block(0, 1, 60000, 1024);
    Matrix Yindex = CIFAR.block(0, 0, 60000, 1);
    Matrix Y;
    Preprocess::oneHot(Yindex, Y);

    const int samples = X.rows();
    const int features = X.cols();
    const int classes = Y.cols();

    X.transposeInPlace();
    Y.transposeInPlace();

    Preprocess::Scaler scaler;
    scaler.inferRange(X);
    scaler.scale(X);

    Matrix w(classes, features);
    MatrixUtil::fillWithRandom(w, 0, 1);

    ComputeEngine CE;

    auto WT = CE.getTensorFromMatrix(w);

    auto W = CE.Var(w);

    // auto X_ = CE.InputVariable();
    // auto Y_ = CE.InputVariable();

    auto X_ = CE.Var(X);
    auto Y_ = CE.Var(Y);

    // auto shuffled = Optimizer::Util::shuffleTensors(CE, {X_, Y_});

    Optimizer::Util::splitMinibatch(CE, X_, Y_, samples, 100, [&W](auto &CE, auto X, auto Y, int batch_samples) {
        auto G = gradientGraph(CE, X, Y, {W}, {batch_samples});

        W = CE.AssignSub(W, CE.Multiply(G, 0.01));
    });


    // auto W_ = CE.Assign(W, NW);

    auto Yhat = CE.Sigmoid(CE.MatMul(W, X_));
    auto NE = leastSquaresLoss(CE, Yhat, Y_);

    const int steps = 10000;
    const int epochs = 1000;
    for (int s = 0; s < epochs; ++s) {
        auto outs = CE.run({}, {}, {NE});
        std::cout << outs[0] / static_cast<Numeric_t>(samples) << std::endl;
    }

    auto Parameters = CE.run({}, {}, {W});

    std::cout << Parameters[0] << std::endl;
}

int main() {
    LR();

    return 0;
}
