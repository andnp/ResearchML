#pragma once
#include "ComputeEngine/matrix.hpp"
#include <vector>

namespace GPUCompute {
namespace Analysis {
    std::vector<Numeric_t> MultiClassClassificationError(const Matrix &P, const Matrix &Y);
    Numeric_t ClassificationError(const Matrix &P, const Matrix &Y);
    Numeric_t Classification1vAllError(const Matrix &P, const Matrix &Y);
    Matrix generateConfusionMatrix(const Matrix &P, const Matrix &Y);
    Numeric_t RMSE(const Matrix &P, const Matrix &Y);
    std::vector<Numeric_t> RMSE_classes(const Matrix &P, const Matrix &Y);
    std::vector<Numeric_t> R2(const Matrix &P, const Matrix &Y);
    std::vector<Numeric_t> MSE(const Matrix &P, const Matrix &Y);
    std::vector<Numeric_t> MAPE(const Matrix &P, const Matrix &Y);
    std::vector<Numeric_t> SMAPE(const Matrix &P, const Matrix &Y);
}}
