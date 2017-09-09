#include <fstream>
#include <cstring>
#include <wordexp.h>

#include "files.hpp"
#include "util/cdash.hpp"

namespace GPUCompute {
namespace Util {
namespace Files {
    std::string getPath(std::string str) {
        wordexp_t exp_results;
        wordexp(str.c_str(), &exp_results, 0);
        return exp_results.we_wordv[0];
    }

    void createFile(std::string str) {
        if (str == "") {
            return;
        }
        str = getPath(str);
        std::vector<std::string> x = _::split(str, '/');
        std::string f = "";
        int offset = 1;
        if (str.back() == '/') offset = 0;
        for (int i = 0; i < x.size() - offset; ++i) {
            f += x[i] + "/";
        }
        if (f.back() == '/') {
            std::string cmd = "mkdir -p " + f;
            std::system(cmd.c_str());
        }
    }
}}}
