#include "preprocess.hpp"
#include "util/cdash.hpp"
#include "ComputeEngine/matrix.hpp"

namespace GPUCompute {
namespace Preprocess {
    void oneHot(const MatrixRef I, Matrix &O) {
        const int rows = I.rows();
        const int cols = I.cols();
        auto maxes = MatrixUtil::colMaxes(I);
        _::add(maxes, 1);
        const int numNewCols = _::sum(maxes);
        int offset = 0;
        O = Matrix::Zero(rows, numNewCols);
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                O(i, offset + I(i, j)) = 1;
            }
            offset += maxes[j];
        }
    }

    std::vector<Matrix> split(Matrix &X, int bins, int axis) {
        Matrix D = axis == 0 ? X : X.transpose();
        int samples = D.rows();
        int features = D.cols();

        int bin_sizes = samples / bins;
        int last_bin = (samples - (bin_sizes * bins)) + bin_sizes;

        std::vector<Matrix> out = {};
        for (int i = 0; i < bins - 1; ++i) {
            out.push_back(D.block(i * bin_sizes, 0, bin_sizes, features));
        }
        out.push_back(D.block(samples - last_bin, 0, last_bin, features));
        return out;
    }

    json Scaler::getDefault() {
        return {
            { "min", 0 },  // min after scaling
            { "max", 1 }  // max after scaling
        };
    }

    Scaler::Scaler() {
        json j = {};
        setConfig(j);
    }

    Scaler::Scaler(json &j) {
        setConfig(j);
    }

    Numeric_t rowMax(const MatrixRef X) {
        return X.array().maxCoeff();
    }

    Numeric_t rowMin(const MatrixRef X) {
        return X.array().minCoeff();
    }

    void Scaler::inferRange(const MatrixRef m) {
        const int rows = m.rows();
        for (int i = 0; i < rows; ++i) {
            auto r = m.row(i);
            const Numeric_t min = rowMin(r);
            const Numeric_t max = rowMax(r);
            scale_min.push_back(min);
            scale_max.push_back(max);
        }
    }

    void Scaler::scale(Matrix &m) {
        const int cols = m.cols();
        const int rows = m.rows();
        for (int i = 0; i < rows; ++i) {
            // scale row between 0 and 1 first
            const Numeric_t divisor = scale_max[i] == scale_min[i] ? 1 : scale_max[i] - scale_min[i];
            m.row(i) = (m.row(i).array() - scale_min[i]) / divisor;
            if (m.row(i).minCoeff() < 0) {
                m.row(i) = (m.row(i) + Matrix::Constant(1.0, cols, 1.0)) / 2.0;
            }
            // then scale to new max
            Numeric_t max = config["max"];
            Numeric_t min = config["min"];
            m.row(i) = m.row(i).array() * (max - min);

            // finally scale to new min
            m.row(i) = m.row(i).array() + min;
        }
    }
}}
