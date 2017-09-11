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

TEST(json, without) {
    json j = R"({
        "one": 1,
        "two": 2,
        "three": 3
    })"_json;

    json o = JSON::without(j, {"one"});
    json e = R"({
        "two": 2,
        "three": 3
    })"_json;

    EXPECT_EQ(o.dump(), e.dump());
}

TEST(json, extendJson) {
    json j1 = R"({
        "one": 1,
        "two": 2
    })"_json;

    json j2 = R"({
        "two": 3,
        "three": 3
    })"_json;

    json e = R"({
        "one": 1,
        "two": 3,
        "three": 3
    })"_json;

    JSON::extendJson(j1, j2);
    EXPECT_EQ(j1.dump(), e.dump());
}

TEST(json, forEach) {
    json j = R"({
        "one": 1,
        "two": 2
    })"_json;

    json o = {};

    JSON::forEach(j, [&o](auto key, auto value) {
        o[key] = value;
    });

    EXPECT_EQ(o.dump(), j.dump());
}
