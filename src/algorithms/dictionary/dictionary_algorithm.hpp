#pragma once
#include "ComputeEngine/matrix.hpp"

namespace GPUCompute {
    class DictionaryAlgorithm {
    public:
        virtual Matrix getDictionary() = 0;
    };
}
