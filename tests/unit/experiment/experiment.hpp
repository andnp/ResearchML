#include "gtest/gtest.h"
#include "gpucompute.hpp"
using namespace GPUCompute;

TEST(experiment, getParameters) {
    json j = R"({
        "param1": [1, 2, 3, 4, 5],
        "param2": [1, 2, 3, 4, 5],
        "param3": [1, 2, 3, 4, 5]
    })"_json;

    json j1 = R"({
        "param1": 1,
        "param2": 1,
        "param3": 1
    })"_json;

    EXPECT_EQ(j1.dump(), ExperimentParser::getParameters(j, 0).dump());

    json j2 = R"({
        "param1": 1,
        "param2": 1,
        "param3": 2
    })"_json;

    EXPECT_EQ(j2.dump(), ExperimentParser::getParameters(j, 25).dump());

    EXPECT_EQ(j1.dump(), ExperimentParser::getParameters(j, 125).dump());
}
