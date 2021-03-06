#include "gtest/gtest.h"
#include "gpucompute.hpp"
using namespace GPUCompute;

// we should be able to iterate over every element in a std::vector
TEST(cdash, forEach) {
    std::vector<int> ints = {1, 2, 3, 4, 5};
    int counter = 1;
    _::forEach(ints, [&counter](int v) {
        EXPECT_TRUE(v == counter++);
    });
}

// we should be able to map a function over every element in a std::vector
TEST(cdash, map) {
    std::vector<int> ints = {1, 2, 3, 4, 5};
    int counter = 1;
    auto out = _::map<int, int>(ints, [&counter](int v) {
        EXPECT_TRUE(v == counter++);
        return v + 1;
    });

    int check = 2;
    _::forEach(out, [&check](int v) {
        EXPECT_TRUE(v == check++);
    });
}

// we should be able to take the mean of a std::vector
TEST(cdash, mean) {
    std::vector<int> ints = {1, 2, 3, 4, 5};
    int counter = 1;
    auto m = _::mean(ints);
    EXPECT_TRUE(m == 3);
}

// we should be able to execute a function n times
TEST(cdash, times) {
    int n = 20;
    auto m = _::times<int>(n, [](int i) {
        return i;
    });

    int counter = 0;
    _::forEach(m, [&counter](int v) {
        EXPECT_TRUE(v == counter++);
    });
}

// we should be able to concatenate multiple vectors into a new one
TEST(cdash, concat) {
    std::vector<int> m = {1, 2, 3};
    std::vector<int> n = {4, 5, 6};
    auto o = _::concat(m, n);

    int counter = 1;
    _::forEach(o, [&counter](int v) {
        EXPECT_TRUE(v == counter++);
    });
}

// we should be able to sum a vector
TEST(cdash, sum) {
    std::vector<int> m = {1, 2, 3, 4, 5};
    auto o = _::sum(m);

    EXPECT_TRUE(o == 15);
}

// we should be able to get the nth entry from the back of a std::vector
TEST(cdash, fromBack) {
    std::vector<int> m = {1, 2, 3, 4, 5};
    auto last = _::fromBack(m, 1);
    EXPECT_TRUE(last == 5);

    auto sndLast = _::fromBack(m, 2);
    EXPECT_TRUE(sndLast == 4);

    auto first = _::fromBack(m, 5);
    EXPECT_TRUE(first == 1);
}

TEST(cdash, fromBack_returnReference) {
    std::vector<int> m = {1, 2, 3, 4, 5};
    _::fromBack(m, 1) = 6;
    EXPECT_EQ(m[4], 6);
}

// we should be able to get the last element from a std::vector
TEST(cdash, last) {
    std::vector<int> m = {1, 2, 3, 4, 5};
    auto last = _::last(m);
    EXPECT_TRUE(last == 5);
}

// we should be able to modify the original vector
TEST(cdash, last_returnReference) {
    std::vector<int> m = {1, 2, 3, 4, 5};
    _::last(m) = 6;
    EXPECT_EQ(m[4], 6);
}

// we should be able to insert at the front of a std::vector
TEST(cdash, insertFront) {
    std::vector<int> m = {3, 4, 5};
    int two = 2;
    _::insertFront(m, two);
    EXPECT_TRUE(m[0] == two);

    int one = 1;
    _::insertFront(m, one);
    EXPECT_TRUE(m[0] == one);
}

// we should be able to add x to every element of std::vector
TEST(cdash, add) {
    std::vector<int> m = {1, 2, 3, 4, 5};
    int x = 10;
    _::add(m, x);

    int counter = 11;
    _::forEach(m, [&counter](int v) {
        EXPECT_TRUE(v == counter++);
    });
}

// we should be able to measure if two floating point numbers are close
TEST(cdash, isClose) {
    double x = 1;
    double y = 2;
    double z = 1 + 1e-7;
    EXPECT_TRUE(_::isClose(x, z));
    EXPECT_FALSE(_::isClose(x, y));
}

// we should be able to drop an element from a std::vector
TEST(cdash, drop) {
    std::vector<int> m = {1, 2, 3, 4, 5};
    EXPECT_TRUE(_::vectorEqual(_::drop(m, 0), {2, 3, 4, 5}));
    EXPECT_TRUE(_::vectorEqual(_::drop(m, 1), {1, 3, 4, 5}));
    EXPECT_TRUE(_::vectorEqual(_::drop(m, 2), {1, 2, 4, 5}));
    EXPECT_TRUE(_::vectorEqual(_::drop(m, 3), {1, 2, 3, 5}));
    EXPECT_TRUE(_::vectorEqual(_::drop(m, 4), {1, 2, 3, 4}));
}

TEST(cdash, in) {
    std::vector<int> v = {1, 2, 3, 4, 5};
    EXPECT_TRUE(_::in(v, 1));
    EXPECT_FALSE(_::in(v, 6));
}

TEST(cdash, contains) {
    std::string str = "hi there friend";
    EXPECT_TRUE(_::contains(str, " th"));
    EXPECT_FALSE(_::contains(str, " their"));
}

TEST(cdash, clone) {
    std::vector<int> a = {1, 2, 3};
    auto b = _::clone(a);
    b.push_back(4);
    a.push_back(5);
    EXPECT_EQ(_::last(a), 5);
    EXPECT_EQ(a.size(), 4);
    EXPECT_EQ(_::last(b), 4);
    EXPECT_EQ(b.size(), 4);
}

TEST(cdash, head) {
    std::vector<int> v = {1, 2, 3, 4, 5};
    auto h = _::head(v, 2);
    EXPECT_EQ(h.size(), 2);
    EXPECT_EQ(h[0], 1);
    EXPECT_EQ(h[1], 2);
}
