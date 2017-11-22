#include "ComputeEngine/matrix.hpp"
#include "util/json.hpp"

namespace GPUCompute {
namespace Transformations {
    void kernelTransformation(Matrix &K, const Matrix &X, const json &parameters);
}}
