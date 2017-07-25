#include "gtest/gtest.h"
#include "gpucompute.hpp"
#include "unit/unit_tests.hpp"

/* ------------- End Read/Write Tests --------------- */

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
