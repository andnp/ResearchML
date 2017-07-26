#pragma once
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

namespace GPUCompute {
    template <class T, class UnaryFunction>
    void forEach(T vector, UnaryFunction f) {
        std::for_each(vector.begin(), vector.end(), f);
    }

    std::vector<std::string> split(const std::string &s, char delim);

    template <typename T>
    T mean(std::vector<T> &v) {
        const int s = v.size();
        T sum = 0.0;
        for (int i = 0; i < s; ++i) {
            sum += v[i];
        }
        return sum / static_cast<T>(s);
    }

    template <typename T, class Func_t>
    std::vector<T> times(int n, Func_t f) {
        std::vector<T> out = {};
        for (int i = 0; i < n; ++i) {
            out.push_back(f(n));
        }
        return out;
    }
}  // namespace GPUCompute
