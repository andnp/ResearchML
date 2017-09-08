#pragma once
#include <vector>
#include <array>
#include "ComputeEngine/matrix.hpp"

namespace GPUCompute {
namespace DataLoader {
class SupervisedData {
public:
    Matrix X;
    Matrix Y;

    inline SupervisedData(MatrixRef X, MatrixRef Y) { this->X = X; this->Y = Y; }
    inline explicit SupervisedData(std::array<Matrix, 2> &data) { this->X = data[0]; this->Y = data[1]; }
};
}}
