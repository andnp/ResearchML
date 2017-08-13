#include "preprocess.hpp"
#include "util/cdash.hpp"

namespace GPUCompute {
    void oneHot(const Matrix &I, Matrix &O) {
        const int rows = I.rows();
        const int cols = I.cols();
        auto maxes = colMaxes(I);
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
}
