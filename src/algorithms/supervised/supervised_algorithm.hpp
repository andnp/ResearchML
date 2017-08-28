#pragma once
#include <vector>
#include <string>

#include "ComputeEngine/matrix.hpp"
#include "algorithms/abstract_algorithm.hpp"

namespace GPUCompute {
    class SupervisedAlgorithm : public Algorithm {
    public:
        explicit SupervisedAlgorithm(json &params);
        // void optimizeCrossValidation(int each, int k);
        // void optimizeCrossValidationStep(int each, int K, int minibatch, int step);
        Matrix computeROC();
        Numeric_t computeAUC(const Matrix &roc);
        virtual Matrix predict(const Matrix &X);
        virtual Numeric_t loss(const MatrixRef X, const MatrixRef Y);
        virtual void optimize(const MatrixRef X, const MatrixRef Y, json opt_params);


        Numeric_t average_test = 0;
        Numeric_t average_train = 0;
        Numeric_t stddev_test = 0;
        Numeric_t stddev_train = 0;

        std::vector<Numeric_t> losses;
        std::vector<Numeric_t> accuracies;

        Numeric_t prediction_threshold = .5;
        Numeric_t test_helper(const MatrixRef Yhat, const MatrixRef Y);

    protected:
        Numeric_t evaluationMethod(const Matrix &P, const Matrix &Y);
    };
}
