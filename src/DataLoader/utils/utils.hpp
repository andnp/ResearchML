#include <vector>
#include "DataLoader/Preprocess/preprocess.hpp"
#include "DataLoader/DataLoader.hpp"
#include "ComputeEngine/matrix.hpp"
#include "util/json.hpp"

namespace GPUCompute {
namespace DataLoader {
namespace Util {
    SupervisedData loadDataset(json options);
    std::vector<SupervisedData> getTestTrain(MatrixRef X, MatrixRef Y, int train, int test);
    std::vector<SupervisedData> getKFoldCV(Matrix &X, Matrix &Y, int k, int fold);
}}}
