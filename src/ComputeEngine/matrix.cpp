#include "matrix.hpp"

#include <vector>
#include <fstream>
#include <string>
#include "util/Files/files.hpp"
#include "util/Random/rand.hpp"

namespace GPUCompute {
    DataType Tensor_t = DT_DOUBLE;

namespace MatrixUtil {
    std::vector<int> getDimensions(std::string file) {
        int col = 0;
        int row = 0;
        std::ifstream indata;
        indata.open(getPath(file));

        std::string line;
        while (std::getline(indata, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            if (row == 0) {
                while (std::getline(lineStream, cell, ',')) {
                    col++;
                }
            }
            row++;
        }
        indata.close();
        return {row, col};
    }

    void readMatrix(std::string file, Matrix &X) {
        file = getPath(file);
        std::vector<int> dims = getDimensions(file);
        const int rows = dims[0];
        const int cols = dims[1];
        X = Matrix(rows, cols);
        std::ifstream indata;
        indata.open(file);

        std::string line;
        int i = 0;
        while (std::getline(indata, line)) {
            int j = 0;
            std::stringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, ',')) {
                X(i, j) = std::stod(cell);
                j++;
            }
            i++;
        }
        indata.close();
    }

    void fillWithRandom(Matrix &M, float mean, float sigma) {
        size_t x = M.rows();
        size_t y = M.cols();

        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                M(i, j) = Random::instance().normal(mean, sigma);
            }
        }
    }

    void fillWithRandom(Matrix &M) {
        fillWithRandom(M, 0, 1);
    }

    std::vector<int> colMaxes(const MatrixRef I) {
        std::vector<int> maxes = {};
        const int cols = I.cols();
        for (int i = 0; i < cols; ++i)
            maxes.push_back(I.col(i).maxCoeff());
        return maxes;
    }

    bool areMatricesEqual(MatrixRef a, MatrixRef b) {
        const Matrix o = zipMatrices(a, b, [](auto v1, auto v2, auto i, auto j) {
            return v1 - v2;
        });
        return o.sum() == 0;
    }
}}
