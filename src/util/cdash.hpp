#pragma once
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

namespace GPUCompute {
namespace _ {
    template <class T, class UnaryFunction>
    void forEach(T vector, UnaryFunction f) {
        std::for_each(vector.begin(), vector.end(), f);
    }

    template <typename T, typename O, class UnaryFunction>
    std::vector<O> map(std::vector<T> vector, UnaryFunction f) {
        std::vector<O> out = {};
        for (int i = 0; i < vector.size(); ++i) {
            out.push_back(
                f(vector[i])
            );
        }
        return out;
    }

    template <typename T1, typename T2, typename O, class Func_t>
    std::vector<O> zip(std::vector<T1> v1, std::vector<T2> v2, Func_t f) {
        std::vector<O> out = {};
        for (int i = 0; i < v1.size(); ++i) {
            out.push_back(
                f(v1[i], v2[i])
            );
        }
        return out;
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
            out.push_back(f(i));
        }
        return out;
    }

    template <class Func_t>
    void times(int n, Func_t f) {
        for (int i = 0; i < n; ++i)
            f(i);
    }

    template <typename T>
    std::vector<T> concat(std::vector<T> &v1, std::vector<T> &v2) {
        std::vector<T> out = {};
        out.insert(out.end(), v1.begin(), v1.end());
        out.insert(out.end(), v2.begin(), v2.end());
        return out;
    }

    template <typename T>
    T sum(const std::vector<T> &v) {
        T total = 0;
        for (int i = 0; i < v.size(); ++i)
            total += v[i];
        return total;
    }

    template <typename T>
    T fromBack(const std::vector<T> &v, const int i = 1) {
        if (i < 1 || v.size() - i < 0) throw "Cannot get index from back of vector";
        return v[v.size() - i];
    }

    template <typename T>
    T last(const std::vector<T> &v) {
        return fromBack(v, 1);
    }

    template <typename T>
    void insertFront(std::vector<T> &v, T x) {
        v.insert(std::begin(v), x);
    }

    template <typename T>
    std::vector<T> drop(const std::vector<T> &v, const int i) {
        std::vector<T> out = {};
        out.insert(out.end(), v.begin(), v.begin() + i);
        out.insert(out.end(), v.begin() + (i+1), v.end());
        return out;
    }

    template <typename T>
    void add(std::vector<T> &v, const T num) {
        for (int i = 0; i < v.size(); ++i)
            v[i] = v[i] + num;
    }

    template <typename T>
    bool isClose(T a, T b) {
        return std::abs(a - b) < 1e-6;
    }

    template <typename T>
    bool vectorEqual(std::vector<T> a, std::vector<T> b) {
        if (a.size() != b.size()) return false;
        for (int i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }
}  // namespace _
}  // namespace GPUCompute
