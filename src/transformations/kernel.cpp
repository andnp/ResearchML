#include "kernel.hpp"
#include "util/cdash.hpp"
#include "ComputeEngine/matrix.hpp"
#include "util/json.hpp"

#include <vector>
#include <algorithm>

namespace GPUCompute {
namespace Transformations {
    int classFromOneHot(const MatrixRef X) {
        const int classes = X.rows();
        if (classes == 1) {
            return X(0, 0);
        }
        int index = -1;
        for (int i = 0; i < classes; ++i) {
            if (X(i, 0) > .5) {
                index = i;
                break;
            }
        }
        return index;
    }

    Matrix getKernelCenters(const Matrix &X, const Matrix &Y, const Matrix &Test, const Matrix &TT, const int centers) {  // NOLINT(runtime/references)
        const size_t rows = X.rows();
        Matrix C(rows, centers);

        const size_t total_samples = X.cols() + Test.cols();
        const int classes = Y.rows() == 1 ? 2 : Y.rows();

        std::vector<Matrix> class_samples(classes);

        for (int i = 0; i < X.cols(); ++i) {
            const int class_num = classFromOneHot(Y.col(i));
            const int cols = class_samples[class_num].cols();
            class_samples[class_num].conservativeResize(X.rows(), cols + 1);
            class_samples[class_num].col(cols) = X.col(i);
        }

        for (int i = 0; i < Test.cols(); ++i) {
            const int class_num = classFromOneHot(TT.col(i));
            const int cols = class_samples[class_num].cols();
            class_samples[class_num].conservativeResize(Test.rows(), cols + 1);
            class_samples[class_num].col(cols) = Test.col(i);
        }

        for (int i = 0; i < classes; ++i) {
            std::cout << class_samples[i].rows() << ", " << class_samples[i].cols() << std::endl;
        }

        // this could be done way faster
        for (int i = 0; i < centers; i++) {
            C.col(i) = class_samples[i % classes].col(floor(static_cast<double>(i) / static_cast<double>(classes)));
        }
        return C;
    }

    Matrix getKernelCenters(const Matrix &X, const Matrix &Test, const int centers) {  // NOLINT(runtime/references)
        const size_t rows = X.rows();
        Matrix C(rows, centers);

        const size_t total_samples = X.cols() + Test.cols();
        const std::vector<int> sample = _::shuffle(total_samples);  // fill with random indexes of X

        // this could be done way faster
        for (int i = 0; i < centers; i++) {
            const int col = sample[i];
            for (int row = 0; row < rows; row++) {
                if (col >= X.cols()) {
                    C(row, i) = Test(row, col - X.cols());
                } else {
                    C(row, i) = X(row, col);
                }
            }
        }
        return C;
    }

    void getCenters(Matrix &O, const Matrix &X, const int centers) {
        const int rows = X.rows();
        O = Matrix(rows, centers);

        auto samples = _::shuffle(X.cols());

        for (int i = 0; i < centers; ++i) {
            const int col = samples[i];
            for (int row = 0; row < rows; ++row) {
                O(row, i) = X(row, col);
            }
        }
    }

    std::vector<Numeric_t> getBandwidths(const MatrixRef C, const double overlap) {
        const size_t numkernel = C.cols();
        const size_t centerfeatures = C.rows();
        std::vector<Numeric_t> bandwidths(numkernel);
        Matrix TC = C;

        for (int i = 0; i < numkernel; i++) {
            double closest = 1e15;
            for (int j = 0; j < numkernel; j++) {
                if (i != j) {
                    const Matrix::ColXpr C1 = TC.col(i);
                    const Matrix::ColXpr C2 = TC.col(j);
                    const double dist = (C1-C2).norm();
                    if (dist < closest) {
                        closest = dist;
                    }
                }
            }
            bandwidths[i] = overlap * closest;
        }
        return bandwidths;
    }

    Numeric_t laplacian_kernel(const Matrix::ColXpr &k1, const Matrix::ColXpr &k2, const Numeric_t sigma) {
        return exp(-1.0 * ((k1-k2).lpNorm<1>() / sigma));
    }

    Numeric_t gaussian_kernel(const Matrix::ColXpr &k1, const Matrix::ColXpr &k2, const Numeric_t sigma) {
        return exp(-1.0 * ((k1-k2).squaredNorm() / (2.0 * sigma*sigma)));
    }

    Numeric_t blockAvg(const Matrix &k, const int x, const int y, const int box_size) {
        Numeric_t avg = 0;
        for (int i = x; i < x + box_size; i++) {
            for (int j = y; j < y + box_size; j++) {
                avg += k(i, j);
            }
        }
        return avg / pow(box_size, 2);
    }

    Numeric_t intersection_kernel(const Matrix::ColXpr &k1, const Matrix::ColXpr &k2) {
        const size_t row = k1.rows();
        const size_t col = k1.cols();
        double sum = 0.0;
        double ele1, ele2;
        for (size_t i = 0; i < row; ++i) {
            for (size_t j = 0; j < col; ++j) {
                ele1 = k1(i, j);
                ele2 = k2(i, j);
                if (ele1 <= ele2)
                    sum += ele1;
                else
                    sum += ele2;
            }
        }
        return sum;
    }

    Numeric_t conv_kernel(const Matrix &k1, const Matrix &k2, const int box_size) {
        // vector<Numeric_t> K1(pow(k1.size1() - box_size, 2));
        // vector<Numeric_t> K2(pow(k2.size1() - box_size, 2));
        // for (int i = 0; i < k1.size1() - box_size; i++) {
        //     for (int j = 0; j < k1.size2() - box_size; j++) {
        //         Numeric_t box_avg1 = blockAvg(k1, i, j, box_size);
        //         Numeric_t box_avg2 = blockAvg(k2, i, j, box_size);
        //         // std::cout << i << " " << j << " " << (j + (k1.size1() / box_size) * i) << std::endl;
        //         K1((j + (k1.size1() - box_size) * i)) = box_avg1;
        //         K2((j + (k1.size1() - box_size) * i)) = box_avg2;
        //     }
        // }
        // return laplacian_kernel(K1, K2, 3.5);
        // return boost::numeric::ublas::inner_prod(K1, K2);
        return 0;
    }

    void kernelTransformation(Matrix &K, const Matrix &X, const json &parameters) {
        int numkernel = parameters["centers"];
        const size_t samples = X.cols();
        int normalize = 0;

        Matrix C;
        getCenters(C, X, numkernel);

        K = Matrix::Zero(numkernel, samples);
        int kernel_type = parameters["kernel"];

        if (kernel_type == -1) {
            K = Matrix(X);
        } else if (kernel_type == 0) {  // gaussian
            Matrix TX = X;
            Matrix TC = C;
            Numeric_t overlap = parameters["overlap"];
            const std::vector<Numeric_t> bandwidths = getBandwidths(C, overlap);
            for (int s = 0; s < samples; s++) {
                const Matrix::ColXpr sample = TX.col(s);
                for (int c = 0; c < numkernel; c++) {
                    const Matrix::ColXpr center = TC.col(c);
                    K(c, s) = gaussian_kernel(center, sample, bandwidths[c]);
                }
            }
            normalize = 1;
        } else if (kernel_type == 1) {  // Intersection Kernel
            Matrix TX = X;
            Matrix TC = C;
            for (int s = 0; s < samples; s++) {
                const Matrix::ColXpr sample = TX.col(s);
                for (int c = 0; c < numkernel; c++) {
                    const Matrix::ColXpr center = TC.col(c);
                    K(c, s) = intersection_kernel(center, sample);
                }
            }
            normalize = 1;
        } else if (kernel_type == 2) {  // Polynomial
            Numeric_t overlap = parameters["overlap"];
            K = C.transpose() * X;
            // K = K.array() + 1.0;
            K = K.array().pow(overlap);

            normalize = 1;
        } else if (kernel_type == 3) {  // Blur Kernel
            // for (int i = 0; i < K.size1(); i++) {
            //     for (int j = 0; j < K.size2(); j++) {
            //         vector<Numeric_t> k1(matrix_column<Matrix<Numeric_t>>(C, i));
            //         vector<Numeric_t> k2(matrix_column<Matrix<Numeric_t>>(X, j));
            //         Matrix<Numeric_t> K1(28, 28);
            //         Matrix<Numeric_t> K2(28, 28);
            //         vectorAsMatrix(k1, K1);
            //         vectorAsMatrix(k2, K2);
            //         K(i, j) = conv_kernel(K1, K2, 5);
            //     }
            // }
            // normalize = 1;
        } else if (kernel_type == 4) {  // Laplacian Kernel
            Numeric_t overlap = parameters["overlap"];
            Matrix TX = X;
            Matrix TC = C;
            const std::vector<Numeric_t> bandwidths = getBandwidths(C, overlap);
            for (int s = 0; s < samples; s++) {
                const Matrix::ColXpr sample = TX.col(s);
                for (int c = 0; c < numkernel; c++) {
                    const Matrix::ColXpr center = TC.col(c);

                    K(c, s) = laplacian_kernel(sample, center, bandwidths[c]);
                }
            }
        }

        if (normalize == 1) {
            K /= K.lpNorm<Eigen::Infinity>();
        }
    }
}}

