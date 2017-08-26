#include <vector>
#include "DataLoader/Preprocess/preprocess.hpp"
#include "ComputeEngine/matrix.hpp"
#include "util/json.hpp"

namespace GPUCompute {
namespace DataLoader {
namespace Util {
    std::vector<Matrix> loadDataset(json options);
}}}
