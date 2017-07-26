#pragma once

#include <random>

namespace GPUCompute {
    class Random {
    public:
        void setSeed(int seed);
        float normal(float mean, float var);
        float uniform(float min, float max);
        std::default_random_engine getEngine();
        static Random& instance();
    private:
        std::default_random_engine r;
        Random();
        int seed;
        static Random ran;
    };
}
