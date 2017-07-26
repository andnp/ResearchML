#include "analysis.hpp"
#include <vector>

namespace GPUCompute {
    Numeric_t RMSE(const Matrix &P, const Matrix &Y) {
        const Numeric_t samples = static_cast<Numeric_t>(P.cols());
        const Numeric_t classes = static_cast<Numeric_t>(P.rows());

        Numeric_t sse = 0;
        for (int i = 0; i < P.rows(); ++i) {
            sse += sqrt((P.row(i) - Y.row(i)).squaredNorm() / samples);
        }

        return sse / classes;
    }

    std::vector<Numeric_t> RMSE_classes(const Matrix &P, const Matrix &Y) {
        const int classes = P.rows();
        std::vector<Numeric_t> rmse(classes);
        for (int i = 0; i < classes; ++i) {
            rmse[i] = RMSE(P.row(i), Y.row(i));
        }
        return rmse;
    }

    std::vector<Numeric_t> R2(const Matrix &P, const Matrix &Y) {
        const int classes = P.rows();
        std::vector<Numeric_t> r2(classes);
        for (int i = 0; i < classes; ++i) {
            Numeric_t mean_tot = Y.row(i).mean();
            Numeric_t ss_tot = (Y.row(i) - Matrix::Constant(1, Y.cols(), mean_tot)).squaredNorm();
            Numeric_t ss_res = (Y.row(i) - P.row(i)).squaredNorm();
            r2[i] = 1.0 - (ss_res / ss_tot);
        }

        return r2;
    }

    std::vector<Numeric_t> MSE(const Matrix &P, const Matrix &Y) {
        const int classes = P.rows();
        const Numeric_t samples = static_cast<Numeric_t>(P.cols());
        std::vector<Numeric_t> mse(classes);
        for (int i = 0; i < classes; ++i) {
            mse[i] = (P.row(i) - Y.row(i)).squaredNorm() / samples;
        }

        return mse;
    }

    std::vector<Numeric_t> MAPE(const Matrix &P, const Matrix &Y) {
        int classes = Y.rows();
        Numeric_t samples = static_cast<Numeric_t>(Y.cols());
        std::vector<Numeric_t> mape(classes);
        for (int i = 0; i < classes; ++i) {
            mape[i] = ((Y.row(i) - P.row(i)).array() / Y.row(i).array()).abs().sum() / samples;
        }
        return mape;
    }

    std::vector<Numeric_t> SMAPE(const Matrix &P, const Matrix &Y) {
        int classes = Y.rows();
        Numeric_t samples = static_cast<Numeric_t>(Y.cols());
        std::vector<Numeric_t> mape(classes);
        for (int i = 0; i < classes; ++i) {
            mape[i] = ((P.row(i) - Y.row(i)).array().abs() / (Y.row(i).array().abs() + P.row(i).array().abs())).sum() / samples;
        }
        return mape;
    }
}

