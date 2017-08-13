#pragma once
#include "ComputeEngine/ComputeEngine.hpp"

namespace GPUCompute {
    template <class Func_t>
    void splitMinibatch(ComputeEngine &CE, Input X, Input Y, int samples, int batch_size, Func_t f) {
        const int num_splits = ceil(samples / batch_size);
        auto Splits = CE.Concat({CE.Fill({num_splits - 1}, batch_size), {-1}}, 0);
        auto X_batches = CE.SplitV(X, Splits, 1, num_splits);
        auto Y_batches = CE.SplitV(Y, Splits, 1, num_splits);
        int accum_samples = 0;
        for (int i = 0; i < num_splits; ++i)
        {
            // assume each batch as batch_size number of samples
            int batch_samples = batch_size;
            // on the last batch we may have fewer samples if not evenly divisible. Account for that here
            if (i == num_splits - 1)
                batch_samples = samples - accum_samples;
            accum_samples += batch_samples;

            f(CE, X_batches[i], Y_batches[i], batch_samples);
        }
    }
}
