#include "analysis.hpp"
#include <vector>

namespace GPUCompute {
    std::vector<Numeric_t> MultiClassClassificationError(const Matrix &P, const Matrix &Y) {
        int classes = Y.rows();
        auto thresh_f = [](Numeric_t x) {
            return x > 0.5 ? 1.0 : 0.0;
        };
        std::vector<Numeric_t> error(classes);
        for (int c = 0; c < classes; ++c) {
            error[c] = ((P.row(c).unaryExpr(thresh_f) - Y.row(c)).array().abs().sum() / static_cast<Numeric_t> (Y.cols()));
        }

        return error;
    }

    Numeric_t ClassificationError(const Matrix &P, const Matrix &Y) {
        auto thresh_f = [](Numeric_t x) {
            return x > 0.5 ? 1.0 : 0.0;
        };

        return ((P.unaryExpr(thresh_f) - Y).array().abs().sum() / (static_cast<Numeric_t> (Y.rows()) * static_cast<Numeric_t> (Y.cols())));
    }

    Numeric_t Classification1vAllError(const Matrix &P, const Matrix &Y) {
        const int numsamples = P.cols();
        const int numfeaturesY = Y.rows();

        Numeric_t prediction_threshold = 0.5;

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

        return 1.0 - (percent / static_cast<Numeric_t> (numsamples));
    }
}
