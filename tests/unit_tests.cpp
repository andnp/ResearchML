#include "gtest/gtest.h"
#include "gpucompute.hpp"
#include "unit/unit_tests.hpp"
#include <stdlib.h>

/* ------------- End Read/Write Tests --------------- */

int main(int argc, char **argv) {
    Random::instance().setSeed(1);
    setenv("TF_CPP_MIN_LOG_LEVEL", "3", 1);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
