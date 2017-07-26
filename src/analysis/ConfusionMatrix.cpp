#include "analysis.hpp"

namespace GPUCompute {
    Matrix generateConfusionMatrix(const Matrix &P, const Matrix &Y) {
        const int classes = P.rows() == 1 ? 2 : P.rows();
        const int samples = P.cols();

        Matrix cf = Matrix::Zero(classes, classes);

        for (int i = 0; i < samples; ++i) {
            if (classes > 2) {
                Matrix::Index p;
                P.col(i).maxCoeff(&p);
                Matrix::Index y;
                Y.col(i).maxCoeff(&y);
                cf(y, p) = cf(y, p) + 1;
            } else {
                const int p = P(0, i) > .5 ? 1 : 0;
                const int y = Y(0, i) > .5 ? 1 : 0;
                cf(y, p) = cf(y, p) + 1;
            }
        }
        return cf;
    }
}
