#include <fstream>
#include "json.hpp"
#include "util/cdash.hpp"

namespace GPUCompute {
namespace JSON {
    std::string getJsonString(const json &js) {
        std::string out = "";
        for (auto &j : json::iterator_wrapper(js)) {
            if (j.value().is_object()) {
                out += getJsonString(j.value());
            } else if (j.value().is_number_integer()) {
                int v = j.value();
                out += std::to_string(v) + ", ";
            } else if (j.value().is_number()) {
                float v = j.value();
                out += std::to_string(v) + ", ";
            } else if (j.value().is_string()) {
                std::string v = j.value();
                out += v + ", ";
            }
        }
        return out;
    }

    void extendJson(json &j1, const json &j2) {
        forEach(j2, [&j1](auto key, auto value) {
            j1[key] = value;
        });
    }

    json without(const json j, std::vector<std::string> keys) {
        json nj = {};
        forEach(j, [&nj, &keys](auto key, auto value) {
            if (!_::in(keys, key)) nj[key] = value;
        });
        return nj;
    }

    json readFile(std::string path) {
        std::ifstream f(path);
        json j;
        j << f;
        f.close();
        return j;
    }

    void JsonConfig::setConfig(json &j) {
        config = {};
        extendJson(config, getDefault());
        extendJson(config, j);
    }

    json JsonConfig::getDefault() { return {}; }
}}
