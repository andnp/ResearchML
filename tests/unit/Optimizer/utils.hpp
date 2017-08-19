#include "gtest/gtest.h"
#include "gpucompute.hpp"
using namespace GPUCompute;

// we should be able to split the dataset into minibatches
TEST(Optimizer_utils, splitMinibatch) {
    ComputeEngine CE;

    int samples = 4;
    Matrix x(3, samples);
    x << 0, 1, 0, 2,
         1, 0, 0, 1,
         0, 0, 1, 4;

    Matrix y(1, samples);
    y << 1, 0, 0, 2;

    auto inputs = CE.InputVariables(2);
    std::vector<tensorflow::Output> x_outs = {};
    std::vector<tensorflow::Output> y_outs = {};
    _::times(samples, [&CE, &x_outs, &y_outs](auto i) {
        x_outs.push_back(CE.Var(Matrix::Zero(3, 1)));
        y_outs.push_back(CE.Var(Matrix::Zero(1, 1)));
    });

    int batch_size = 1;
    int count = 0;
    Optimizer::Util::splitMinibatch(CE, inputs[0], inputs[1], samples, batch_size, [&count, &x_outs, &y_outs](auto &CE, auto X, auto Y, int batch_samples) {
        x_outs[count] = CE.Add(x_outs[count], X);
        y_outs[count] = CE.Add(y_outs[count], Y);
        count++;
    });

    auto outputs = CE.run(inputs, {x, y}, _::concat(x_outs, y_outs));

    Matrix ox(3, samples);
    ox << outputs[0], outputs[1], outputs[2], outputs[3];

    Matrix oy(1, samples);
    oy << outputs[4], outputs[5], outputs[6], outputs[7];

    EXPECT_TRUE(MatrixUtil::areMatricesEqual(x, ox));
    EXPECT_TRUE(MatrixUtil::areMatricesEqual(y, oy));
}

// we should be able to split the dataset into minibatches with the number of samples not evenly divisible by batch size
TEST(Optimizer_utils, splitMinibatch_notDivisible) {
    ComputeEngine CE;

    int samples = 4;
    Matrix x(3, samples);
    x << 0, 1, 0, 2,
         1, 0, 0, 1,
         0, 0, 1, 4;

    Matrix y(1, samples);
    y << 1, 0, 0, 2;

    auto inputs = CE.InputVariables(2);
    std::vector<tensorflow::Output> x_outs = {};
    std::vector<tensorflow::Output> y_outs = {};

    x_outs.push_back(CE.Var(Matrix::Zero(3, 3)));
    y_outs.push_back(CE.Var(Matrix::Zero(1, 3)));

    x_outs.push_back(CE.Var(Matrix::Zero(3, 1)));
    y_outs.push_back(CE.Var(Matrix::Zero(1, 1)));

    int batch_size = 3;
    int count = 0;
    Optimizer::Util::splitMinibatch(CE, inputs[0], inputs[1], samples, batch_size, [&count, &x_outs, &y_outs](auto &CE, auto X, auto Y, int batch_samples) {
        if (count == 0) EXPECT_EQ(batch_samples, 3);
        else if (count == 1) EXPECT_EQ(batch_samples, 1);
        x_outs[count] = CE.Add(x_outs[count], X);
        y_outs[count] = CE.Add(y_outs[count], Y);
        count++;
    });

    auto outputs = CE.run(inputs, {x, y}, _::concat(x_outs, y_outs));

    Matrix ox(3, samples);
    ox << outputs[0], outputs[1];

    Matrix oy(1, samples);
    oy << outputs[2], outputs[3];

    EXPECT_TRUE(MatrixUtil::areMatricesEqual(x, ox));
    EXPECT_TRUE(MatrixUtil::areMatricesEqual(y, oy));
}

// we should be able to shuffle multiple tensors in the same way (say shuffle X and Y such that rows are still aligned across these tensors)
TEST(Optimizer_utils, shuffleTensors) {
    ComputeEngine CE;

    Matrix x(2, 3);
    x << 1, 2, 3,
         4, 5, 6;

    auto input = CE.InputVariable();
    auto shuffled = Optimizer::Util::shuffleTensors(CE, {input});

    auto output = CE.run({input}, {x}, shuffled);

    std::cout << output[0] << std::endl;
}
