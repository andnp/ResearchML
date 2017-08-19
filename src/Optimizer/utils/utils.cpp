#include "utils.hpp"
#include <vector>
#include "util/cdash.hpp"
#include "util/Random/rand.hpp"

namespace GPUCompute {
namespace Optimizer {
namespace Util {
    std::vector<TFNode> shuffleTensors(ComputeEngine &CE, std::vector<TFNode> Tensors) {
        int seed = Random::uniformInt(0, 1e8);
        auto shuffled = _::map<TFNode>(Tensors, [&CE, seed](TFNode T) {
            return CE.Transpose(CE.RandomShuffle(CE.Transpose(T), seed));
        });
        return shuffled;
    }
}}}
