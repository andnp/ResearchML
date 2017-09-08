#include "utils.hpp"
#include <string>
#include "util/cdash.hpp"

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

    SupervisedData loadDataset(json user_options) {
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

    std::vector<SupervisedData> getTestTrain(MatrixRef X, MatrixRef Y, int train, int test) {
        Matrix x = X.block(0, 0, X.rows(), train);
        Matrix y = Y.block(0, 0, Y.rows(), train);
        Matrix t = X.block(train, 0, X.rows(), test);
        Matrix tt = Y.block(train, 0, Y.rows(), test);

        return {{x, y}, {t, tt}};
    }

    std::vector<SupervisedData> getKFoldCV(Matrix &X, Matrix &Y, int k, int fold) {
        auto x_bins = Preprocess::split(X, k, 1);
        auto y_bins = Preprocess::split(Y, k, 1);

        auto x_dropped = _::drop(x_bins, fold);
        auto y_dropped = _::drop(y_bins, fold);

        auto x_fold = x_bins[fold];
        auto y_fold = y_bins[fold];

        auto train = SupervisedData(MatrixUtil::stackCols(x_dropped), MatrixUtil::stackCols(y_dropped));

        auto test = SupervisedData(x_fold, y_fold);

        return {train, test};
    }
}}}
