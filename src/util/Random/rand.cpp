#include "rand.hpp"

#include <random>
#include <iostream>

namespace GPUCompute {
    Random Random::ran;

    void Random::setSeed(int seed) {
        r.seed(seed);
        Random::seed = seed;
    }

    float Random::normal(float mean, float var) {
        auto r = Random::instance().getEngine();
        std::normal_distribution<float> dist(mean, var);
        return dist(*r);
    }

    float Random::uniform(float min, float max) {
        auto r = Random::instance().getEngine();
        std::uniform_real_distribution<float> dist(min, max);
        return dist(*r);
    }

    int Random::uniformInt(int min, int max) {
        auto r = Random::instance().getEngine();
        std::uniform_int_distribution<int> dist(min, max);
        return dist(*r);
    }

    std::default_random_engine* Random::getEngine() {
        return &r;
    }

    Random& Random::instance() {
        return ran;
    }

    Random::Random() {
        r.seed(time(0));
    }
}
