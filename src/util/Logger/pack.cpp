#include "pack.hpp"
#include <fstream>

#include "util/json.hpp"
#include "util/cdash.hpp"
#include "util/Files/files.hpp"
#include "util/Logger/Logger.hpp"

namespace GPUCompute {
namespace Packer {
static void touch(std::string file) {
    std::ofstream outfile(file);
    outfile.close();
}

static std::string parseInterpolations(std::string file, const Experiment &e) {
    if (file == "_project_name_") {
        return e.config["project_name"];
    } else if (file == "_dataset_") {
        return e.config["dataset"];
    } else if (file == "_algorithm_") {
        return e.config["algorithm"];
    } else if (file == "_experiment_name_") {
        return e.config["name"];
    } else if (file == "_experiment_") {
        return "experiment.json";
    } else if (file == "_results_") {
        return "results/run-" + std::to_string(e.index);
    } else if (file == "_parameters_") {
        std::string headers = csvHeader(e.config["parameters"]);
        std::string values = JSON::getJsonString(e.config["parameters"]);
        auto h_split = _::split(headers.substr(0, headers.length() - 2), ',');
        auto v_split = _::split(values.substr(0, values.length() - 2), ',');
        std::string out = "";
        for (int i = 0; i < h_split.size(); ++i) {
            out += _::replaceAll(h_split[i], " ", "") + "-" +
                   _::replaceAll(v_split[i], " ", "") + "_";
        }
        return out.substr(0, out.length() - 1);
    } else {
        return file;
    }
}

static void addPath(json js, std::string root, const Experiment &e) {
    // check for files at this level
    if (!js["files"].is_null()) {
        for (std::string f : js["files"]) {
            std::string path = root + parseInterpolations(f, e);
            Util::Files::createFile(path);
            touch(path);
            if (f == "_experiment_") {
                std::ofstream of(path);
                of << e.config.dump(4);
                of.close();
            }
        }
    }

    // check for subdirectories
    if (!js["directories"].is_null()) {
        for (auto &j : json::iterator_wrapper(js["directories"])) {
            std::string path = root + parseInterpolations(j.key(), e) + "/";
            Util::Files::createFile(path);
            addPath(j.value(), path, e);
            if (j.key() == "_results_") {
                Logger::addFilepath(path);
            }
        }
    }
}

void pack(const Experiment &e) {
    json lib_settings = JSON::readFile("../lib_settings.json");

    if (!lib_settings["manifest"].is_null()) {
        addPath(lib_settings["manifest"], "", e);
    }
}
}};  // namespace Packer

/* Example Manifest
"manifest": {
    "files": [
        "Description",
        "manifest.json"
    ],
    "directories": {
        "_algorithm_": {
            "files": ["_results_"],
        }
    }
}

*/
