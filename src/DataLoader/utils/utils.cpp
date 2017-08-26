#include "utils.hpp"
#include <string>

namespace GPUCompute {
namespace DataLoader {
namespace Util {
    json getLoaderDefaults(json &j) {
        json defaults = {
            {"target_char", 0},
            {"one_hot", true},
            {"scale_data", true},
            {"is_row_major", true}
        };
        JSON::extendJson(defaults, j);
        return defaults;
    }

    std::vector<Matrix> loadDataset(json user_options) {
        json options = getLoaderDefaults(user_options);
        std::string filepath = options["path"];
        bool use_one_hot = options["one_hot"];
        bool scale_data = options["scale_data"];
        bool should_transpose = options["is_row_major"];
        int target_position = options["target_char"];


        Matrix data;
        MatrixUtil::readMatrix(filepath, data);

        int total_samples = data.rows();
        int total_features = data.cols();

        Matrix X = data.block(0, 1, total_samples, total_features - 1);
        Matrix Yindex = data.block(0, 0, total_samples, 1);
        Matrix Y;
        if (use_one_hot) {
            Preprocess::oneHot(Yindex, Y);
        } else {
            Y = Yindex;
        }

        const int samples = X.rows();
        const int features = X.cols();
        const int classes = Y.cols();

        if (should_transpose) {
            X.transposeInPlace();
            Y.transposeInPlace();
        }

        if (scale_data) {
            Preprocess::Scaler scaler;
            scaler.inferRange(X);
            scaler.scale(X);
        }

        return {X, Y};
    }
}}}
