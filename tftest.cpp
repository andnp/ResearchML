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

TFNode gradientGraph(ComputeEngine &CE, TFNode X, TFNode Y, std::vector<TFNode> Parameters, json dims) {
    TFNode W = Parameters[0];
    int samples = dims["samples"];
    // compute Y - (sig(W * X));
    auto E = Loss::logisticError(CE, X, Y, W);
    auto G = CE.Div(CE.MatMul(E, X, MatMul::TransposeB(true)), static_cast<Numeric_t>(samples));
    return CE.Add(CE.Multiply(W, 2.0 * 0.0001), G);
}

TFNode lossGraph(ComputeEngine &CE, TFNode X, TFNode Y, std::vector<TFNode> Parameters, json dims) {
    TFNode W = Parameters[0];
    int samples = dims["samples"];
    auto Yhat = CE.Sigmoid(CE.MatMul(W, X));
    auto NE = Loss::leastSquaresLoss(CE, Yhat, Y);
    return CE.Div(NE, static_cast<Numeric_t>(samples));
}

template <class GradientFunc_t, class LossFunc_t>
void optimizeGradientDescent(MatrixRef X, MatrixRef Y, std::vector<MatrixRef> Parameters, json opt_params, GradientFunc_t getGradient, LossFunc_t getLoss) {
    ComputeEngine CE;
    int batch_size = opt_params["batch_size"];
    Numeric_t threshold = opt_params["threshold"];
    Numeric_t stepsize = opt_params["stepsize"];
    int samples = X.cols();

    auto data = CE.InputVariables(2);

    std::vector<TFNode> P = _::map<MatrixRef, TFNode>(Parameters, [&CE](auto x) {
        return CE.Var(x);
    });

    auto shuffled = Optimizer::Util::shuffleTensors(CE, data);
    Optimizer::Util::splitMinibatch(CE, shuffled[0], shuffled[1], samples, batch_size, [&P, stepsize, getGradient](auto &CE, auto X, auto Y, int batch_samples) {
        json dims = {{"samples", batch_samples}};
        auto G = getGradient(CE, X, Y, P, dims);

        P[0] = CE.AssignSub(P[0], CE.Multiply(G, stepsize));
    });

    auto NE = getLoss(CE, data[0], data[1], P, {{"samples", samples}});

    Numeric_t last_loss = 1e10;
    Numeric_t loss = 0;
    while (std::abs(last_loss - loss) > threshold) {
        last_loss = loss;
        auto outs = CE.run(data, {X, Y}, {NE});
        loss = outs[0](0, 0);
        std::cout << outs[0] << std::endl;
    }

    auto p = CE.run(data, {X, Y}, P);

    std::cout << p[0] << std::endl;
}

void LR() {
    // auto data = DataLoader::Util::loadDataset({
    //     {"path", "~/Projects/research/ml_data/cifar10.csv"},
    // });

    Matrix x(4, 4);
    Matrix y(1, 4);

    x << 1, 1, 0, 1,
         0, 0, 1, 1,
         1, 0, 1, 0,
         0, 1, 0, 1;

    y << 1, 1, 0, 1;

    std::vector<Matrix> data = {
        x,  // data matrix
        y   // target matrix
    };

    int classes = data[1].rows();
    int features = data[0].rows();

    Matrix w = MatrixUtil::getRandomMatrix(classes, features, 0, 0.01);

    optimizeGradientDescent(data[0], data[1], {w}, {
        {"batch_size", 1},
        {"threshold", 1e-8},
        {"stepsize", 0.05}
    }, gradientGraph, lossGraph);
}

int main() {
    LR();

    return 0;
}
