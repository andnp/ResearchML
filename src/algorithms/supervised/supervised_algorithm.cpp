#include "supervised_algorithm.hpp"
#include "ComputeEngine/matrix.hpp"
#include "util/Logger/Logger.hpp"
#include "analysis/analysis.hpp"
#include "util/cdash.hpp"

namespace GPUCompute {
    Numeric_t SupervisedAlgorithm::evaluationMethod(const Matrix &P, const Matrix &Y) {
        if (task == "classification") {
            return test_helper(P, Y);
        } else if (task == "regression") {
            std::vector<Numeric_t> mse = MSE(P, Y);
            return mean(mse);
        } else if (task == "multitask_classification") {
            return 1.0 - ClassificationError(P, Y);
        }
        return test_helper(P, Y);
    }

    // void SupervisedAlgorithm::optimizeCrossValidation(int each, int K) {
    //     int num_sweeps = Algorithm::getNumberOfSweeps();
    //     float best_value = -1;
    //     int best_settings = -1;
    //     Algorithm::dat->createKFoldCV(K);
    //     for (int i = 0; i < num_sweeps; ++i) {
    //         float average_accuracy_test = 0.0;
    //         setSweepParameters(i);
    //         for (int k = 0; k < K; ++k) {
    //             Algorithm::dat->setNthTrainingSet(k);
    //             setup();
    //             reset();
    //             optimizeBatch(Algorithm::dat->getTrainingData().rows(), each);
    //             Matrix X = Algorithm::dat->getValidationData();
    //             Matrix P = predict(X);
    //             average_accuracy_test += evaluationMethod(P, Algorithm::dat->getValidationTargets());
    //             // printProgress(static_cast<float>(++complete)/static_cast<float>(K * num_sweeps));
    //         }
    //         if (average_accuracy_test > best_value) {
    //             best_value = average_accuracy_test;
    //             best_settings = i;
    //         }
    //     }
    //     Algorithm::dat->deleteCV();
    //     setup();
    //     reset();
    //     setSweepParameters(best_settings);
    //     Algorithm::optimizeRamp(Algorithm::dat->getTrainingData().rows(), each);
    //     average_test = test();
    //     average_train = TrainAccuracy();
    // }

    // void SupervisedAlgorithm::optimizeCrossValidationStep(int each, int K, int minibatch, int step) {
    //     Algorithm::dat->createKFoldCV(K);
    //     float average_accuracy_test = 0.0;
    //     float average_accuracy_train = 0.0;
    //     setSweepParameters(step);
    //     for (int k = 0; k < K; ++k) {
    //         Algorithm::dat->setNthTrainingSet(k);
    //         setup();
    //         reset();
    //         optimizeBatch(minibatch, each);
    //         Matrix X = Algorithm::dat->getValidationData();
    //         Matrix P = predict(X);
    //         average_accuracy_test += evaluationMethod(P, Algorithm::dat->getValidationTargets());
    //         // printProgress(static_cast<float>(k+1)/static_cast<float>(K));

    //         X = Algorithm::dat->getTrainingData();
    //         P = predict(X);
    //         average_accuracy_train += evaluationMethod(P, Algorithm::dat->getTrainingTargets());
    //     }
    //     Algorithm::dat->deleteCV();
    //     average_test = average_accuracy_test / static_cast<float>(K);
    //     average_train = average_accuracy_train / static_cast<float>(K);
    //     // std::cout << step << ", " << average_accuracy_test / static_cast<float>(K) << std::endl;
    // }

    float SupervisedAlgorithm::test_helper(const MatrixRef P, const MatrixRef Y) {
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

        double percent = 0.0;

        for (int i = 0; i < numsamples; i++) {
            if (targets[i] == results[i]) percent++;
        }

        return percent / static_cast<double> (numsamples);
    }

    int positiveClassCount(const Matrix &Y) {
        int count = 0;
        for (int i = 0; i < Y.cols(); ++i) {
            count += std::round(Y(0, i));
        }
        return count;
    }

    int falsePositives(const Matrix &P, const Matrix &Y, float prediction_threshold) {
        int count = 0;
        for (int i = 0; i < Y.cols(); ++i) {
            if (Y(0, i) == 0 && P(0, i) > prediction_threshold) {
                count++;
            }
        }
        return count;
    }

    int truePositives(const Matrix &P, const Matrix &Y, float prediction_threshold) {
        int count = 0;
        for (int i = 0; i < Y.cols(); ++i) {
            if (Y(0, i) == 1 && P(0, i) > prediction_threshold) {
                count++;
            }
        }
        return count;
    }

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

    float SupervisedAlgorithm::computeAUC(const Matrix &roc) {
        const int precision = roc.rows() - 1;
        float sum = 0.0;
        for (int i = 0; i < precision; ++i) {
            const float width = roc(i, 0) - roc(i+1, 0);
            const float height = (roc(i, 1) + roc(i+1, 1)) / 2.0;
            const float sliver = width*height;
            sum += sliver;
        }
        return sum;
    }

    float SupervisedAlgorithm::loss() {
        throw std::invalid_argument("Uh oh. You didn't implement this for this algorithm. Silly");
        return 0.0;
    }

    Matrix SupervisedAlgorithm::predict(const Matrix &X) {  // NOLINT(runtime/references)
        throw std::invalid_argument("Oopsies!! This isn't implemented yet! Probably shouldn't be using it. (Or take some time and implement it!)");
        return Matrix(0, 0);
    }

    float SupervisedAlgorithm::TrainAccuracy() {
        throw std::invalid_argument("This doesn't work for this algorithm yet. Sorry!");
        return 0.0;
    }

    float SupervisedAlgorithm::loss(const MatrixRef X, const MatrixRef Y) {
        throw std::invalid_argument("loss(X, Y) doesn't work for this algorithm yet. Sorry!");
        return 0.0;
    }
}
