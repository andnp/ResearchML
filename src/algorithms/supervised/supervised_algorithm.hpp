#pragma once
#include <vector>
#include <string>

#include "ComputeEngine/matrix.hpp"
#include "algorithms/abstract_algorithm.hpp"

namespace GPUCompute {
    class SupervisedAlgorithm : public Algorithm {
    public:
        // void optimizeCrossValidation(int each, int k);
        // void optimizeCrossValidationStep(int each, int K, int minibatch, int step);
        virtual float test() = 0;
        virtual float TrainAccuracy() = 0;
        Matrix computeROC();
        float computeAUC(const Matrix &roc);
        virtual Matrix predict(const Matrix &X);

        virtual float loss();
        virtual float loss(const MatrixRef X, const MatrixRef Y);

        float average_test = 0;
        float average_train = 0;
        float stddev_test = 0;
        float stddev_train = 0;

        std::vector<Numeric_t> losses;
        std::vector<Numeric_t> accuracies;

        float prediction_threshold = .5;

    protected:
        float test_helper(const MatrixRef Yhat, const MatrixRef Y);
        Numeric_t evaluationMethod(const Matrix &P, const Matrix &Y);
    };
}
