#pragma once
#include "ComputeEngine/ComputeEngine.hpp"
#include "ComputeEngine/matrix.hpp"
#include "Optimizer/utils/utils.hpp"
#include "util/json.hpp"
#include "util/Logger/Logger.hpp"

namespace GPUCompute {
namespace Optimizer {
    template <class GradientFunc_t, class LossFunc_t>
    std::vector<Matrix> optimizeGradientDescent(MatrixRef X, MatrixRef Y, std::vector<Matrix> Parameters, json opt_params, GradientFunc_t getGradient, LossFunc_t getLoss) {
        ComputeEngine CE;
        int batch_size = opt_params["batch_size"];
        Numeric_t threshold = opt_params["threshold"];
        Numeric_t stepsize = opt_params["stepsize"];
        int samples = X.cols();

        if (samples / batch_size > 500) Logger::warn() << "Number of minibatches is quite large. This may cause algorithm to spend unnecessary time creating computational graph." << std::endl;

        auto data = CE.InputVariables(2);

        std::vector<TFNode> P = _::map<Matrix, TFNode>(Parameters, [&CE](auto x) {
            return CE.Var(x);
        });

        auto shuffled = Optimizer::Util::shuffleTensors(CE, data);
        Optimizer::Util::splitMinibatch(CE, shuffled[0], shuffled[1], samples, batch_size, [&P, stepsize, getGradient](auto &CE, auto X, auto Y, int batch_samples) {
            json dims = {{"samples", batch_samples}};
            auto G = getGradient(CE, X, Y, P, dims);

            for (int i = 0; i < P.size(); ++i) {
                P[i] = CE.AssignSub(P[i], CE.Multiply(G[i], stepsize));
            }
        });

        auto NE = getLoss(CE, data[0], data[1], P, {{"samples", samples}});

        Numeric_t last_loss = 1e10;
        Numeric_t loss = 0;
        while (std::abs(last_loss - loss) > threshold) {
            last_loss = loss;
            auto outs = CE.run(data, {X, Y}, {NE});
            loss = outs[0](0, 0);
            Logger::aux("loss.csv") << loss << std::endl;
        }

        return CE.run(data, {X, Y}, P);
    }
}}
