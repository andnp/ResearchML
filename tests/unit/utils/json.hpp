#include "gtest/gtest.h"
#include "gpucompute.hpp"
using namespace GPUCompute;

TEST(json, length_numbers) {
    json j = {1, 2, 3, 4, 5};
    EXPECT_EQ(j.size(), 5);
}

TEST(json, length_strings) {
    json j = {"hi", "there", "friend"};
    EXPECT_EQ(j.size(), 3);
}

TEST(json, length_objects) {
    json j = {{
        {"one", 1},
        {"two", 2}
    },
    {
        {"one", 2},
        {"two", 3}
    }};
    EXPECT_EQ(j.size(), 2);
}
