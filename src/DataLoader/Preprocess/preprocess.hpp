#pragma once

#include <vector>

#include "ComputeEngine/matrix.hpp"
#include "util/json.hpp"

namespace GPUCompute {
namespace Preprocess {
    void oneHot(const MatrixRef I, Matrix &O);
    std::vector<Matrix> split(MatrixRef X, int bins, int axis = 0);

    class Scaler : public JSON::JsonConfig {
        std::vector<Numeric_t> scale_min = {}; // min observation from dataset for each row
        std::vector<Numeric_t> scale_max = {}; // max observation from dataset for each row
        json getDefault() override;
    public:
        Scaler();
        explicit Scaler(json &j);
        void inferRange(const MatrixRef m);
        void scale(Matrix &m);
    };
}}
