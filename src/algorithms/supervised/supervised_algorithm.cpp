#include "supervised_algorithm.hpp"
#include "ComputeEngine/matrix.hpp"
#include "util/Logger/Logger.hpp"
#include "analysis/analysis.hpp"
#include "util/cdash.hpp"

namespace GPUCompute {
    SupervisedAlgorithm::SupervisedAlgorithm(json &params) : Algorithm(params) {};

    Numeric_t SupervisedAlgorithm::evaluationMethod(const Matrix &P, const Matrix &Y) {
        std::string task = config["task"];
        if (task == "classification") {
            return test_helper(P, Y);
        } else if (task == "regression") {
            std::vector<Numeric_t> mse = Analysis::MSE(P, Y);
            return _::mean(mse);
        } else if (task == "multitask_classification") {
            return 1.0 - Analysis::ClassificationError(P, Y);
        }
        return test_helper(P, Y);
    }

    // TODO: This needs to be moved to the analysis folder instead
    Numeric_t SupervisedAlgorithm::test_helper(const MatrixRef P, const MatrixRef Y) {
        const int numsamples = P.cols();
        const int numfeaturesY = Y.rows();

        std::vector<int> results(numsamples);
        std::vector<int> targets(numsamples);

        if (numfeaturesY > 1) {
            for (int j = 0; j < numsamples; j++) {
                Matrix::Index loc;
                P.col(j).maxCoeff(&loc);
                results[j] = loc;
            }

            for (int j = 0; j < numsamples; j++) {
                Matrix::Index loc;
                Y.col(j).maxCoeff(&loc);
                targets[j] = loc;
            }
        } else {
            for (int i = 0; i < numsamples; ++i) {
                results[i] = P(0, i) > prediction_threshold ? 1 : 0;
                targets[i] = Y(0, i);
            }
        }

        Numeric_t percent = 0.0;

        for (int i = 0; i < numsamples; i++) {
            if (targets[i] == results[i]) percent++;
        }

        return percent / static_cast<Numeric_t> (numsamples);
    }

    int positiveClassCount(const Matrix &Y) {
        int count = 0;
        for (int i = 0; i < Y.cols(); ++i) {
            count += std::round(Y(0, i));
        }
        return count;
    }

    int falsePositives(const Matrix &P, const Matrix &Y, Numeric_t prediction_threshold) {
        int count = 0;
        for (int i = 0; i < Y.cols(); ++i) {
            if (Y(0, i) == 0 && P(0, i) > prediction_threshold) {
                count++;
            }
        }
        return count;
    }

    int truePositives(const Matrix &P, const Matrix &Y, Numeric_t prediction_threshold) {
        int count = 0;
        for (int i = 0; i < Y.cols(); ++i) {
            if (Y(0, i) == 1 && P(0, i) > prediction_threshold) {
                count++;
            }
        }
        return count;
    }

    // TODO: This needs to be moved to the analysis folder instead
    Matrix SupervisedAlgorithm::computeROC() {
        // const Matrix TestTarget = dat->getTestTargets();
        // Matrix Test = dat->getTestData();
        // const int input_bias = parameters["input_bias"].is_null() ? 0 : static_cast<int>(parameters["input_bias"]);
        // const int pos = positiveClassCount(TestTarget);
        // const int neg = TestTarget.cols() - pos;
        // Matrix roc(101, 2);
        // const Matrix P = predict(Test.block(0, 0, Test.rows() -input_bias, Test.cols()));
        // for (int i = 0; i < 101; ++i) {
        //     prediction_threshold = static_cast<float>(i) / 100.0;
        //     const float fpr = falsePositives(P, TestTarget, prediction_threshold) / static_cast<float>(neg);
        //     const float tpr = truePositives(P, TestTarget, prediction_threshold) / static_cast<float>(pos);
        //     roc(i, 0) = fpr;
        //     roc(i, 1) = tpr;
        // }
        // return roc;
    }

    // TODO: This needs to be moved to the analysis folder instead
    Numeric_t SupervisedAlgorithm::computeAUC(const Matrix &roc) {
        const int precision = roc.rows() - 1;
        Numeric_t sum = 0.0;
        for (int i = 0; i < precision; ++i) {
            const Numeric_t width = roc(i, 0) - roc(i+1, 0);
            const Numeric_t height = (roc(i, 1) + roc(i+1, 1)) / 2.0;
            const Numeric_t sliver = width*height;
            sum += sliver;
        }
        return sum;
    }

    Matrix SupervisedAlgorithm::predict(const Matrix &X) {  // NOLINT(runtime/references)
        throw std::invalid_argument("Oopsies!! This isn't implemented yet! Probably shouldn't be using it. (Or take some time and implement it!)");
        return Matrix(0, 0);
    }

    Numeric_t SupervisedAlgorithm::loss(const MatrixRef X, const MatrixRef Y) {
        throw std::invalid_argument("loss(X, Y) doesn't work for this algorithm yet. Sorry!");
        return 0.0;
    }

    void SupervisedAlgorithm::optimize(const MatrixRef X, const MatrixRef Y, json opt_params) {
        throw "optimize has not yet been implemented";
    }
}
