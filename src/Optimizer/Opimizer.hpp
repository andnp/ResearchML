#pragma once
#include "ComputeEngine/ComputeEngine.hpp"
#include "ComputeEngine/matrix.hpp"
#include "Optimizer/utils/utils.hpp"
#include "util/json.hpp"
#include "util/Logger/Logger.hpp"

namespace GPUCompute {
namespace Optimizer {
    inline json mergeDefault(json &j) {
        json config = {};
        JSON::extendJson(config, {
            {"threshold", 1.0e-6},
            {"max_steps", -1}
        });
        JSON::extendJson(config, j);
        return config;
    }
    // -------------------
    // Optimizer Framework
    // -------------------
    template <class GradientFunc_t, class LossFunc_t, class OptFunc_t>
    std::vector<Matrix> optimize(ComputeEngine &CE, MatrixRef X, MatrixRef Y, std::vector<Matrix> Parameters, json opt_params, GradientFunc_t getGradient, LossFunc_t getLoss, OptFunc_t opt) {
        opt_params = mergeDefault(opt_params);
        int batch_size = opt_params["batch_size"];
        Numeric_t threshold = opt_params["threshold"];
        int max_steps = opt_params["max_steps"];
        int samples = X.cols();

        if (samples / batch_size > 500) Logger::warn() << "Number of minibatches is quite large. This may cause algorithm to spend unnecessary time creating computational graph." << std::endl;

        auto data = CE.InputVariables(2);

        std::vector<TFNode> P = _::map<Matrix, TFNode>(Parameters, [&CE](auto x) {
            return CE.Var(x);
        });

        auto shuffled = Optimizer::Util::shuffleTensors(CE, data);
        Optimizer::Util::splitMinibatch(CE, shuffled[0], shuffled[1], samples, batch_size, [&P, getGradient, opt](auto &CE, auto X, auto Y, int batch_samples) {
            json dims = {{"samples", batch_samples}};
            auto G = getGradient(CE, X, Y, P, dims);

            for (int i = 0; i < P.size(); ++i) {
                P[i] = opt(P[i], G[i], i);
                // P[i] = CE.ApplyGradientDescent(P[i], stepsize, G[i]);
            }
        });

        auto NE = getLoss(CE, data[0], data[1], P, {{"samples", samples}});

        Numeric_t last_loss = 1e10;
        Numeric_t loss = 0;
        int step = 0;
        while (std::abs(last_loss - loss) > threshold &&
              (max_steps > -1 && step < max_steps)
        ) {
            last_loss = loss;
            auto outs = CE.run(data, {X, Y}, {NE});
            loss = outs[0](0, 0);
            Logger::aux("loss.csv") << loss << std::endl;
            step++;
        }

        return CE.run(data, {X, Y}, P);
    }

    // -------------------
    // Gradient Optimizers
    // -------------------

    template <class GradientFunc_t, class LossFunc_t>
    std::vector<Matrix> optimizeGradientDescent(MatrixRef X, MatrixRef Y, std::vector<Matrix> Parameters, json opt_params, GradientFunc_t getGradient, LossFunc_t getLoss) {
        ComputeEngine CE;
        Numeric_t stepsize = opt_params["stepsize"];
        return optimize(CE, X, Y, Parameters, opt_params, getGradient, getLoss, [&CE, stepsize](TFNode P, TFNode G, int i) {
            return CE.ApplyGradientDescent(P, stepsize, G);
        });
    };

    template <class GradientFunc_t, class LossFunc_t>
    std::vector<Matrix> optimizeAdadelta(MatrixRef X, MatrixRef Y, std::vector<Matrix> Parameters, json opt_params, GradientFunc_t getGradient, LossFunc_t getLoss) {
        ComputeEngine CE;
        Numeric_t rho = opt_params["rho"];
        Numeric_t epsilon = opt_params["epsilon"];

        std::vector<TFNode> EG = _::map<Matrix, TFNode>(Parameters, [&CE](auto x) {
            Matrix z = Matrix::Zero(x.rows(), x.cols());
            return CE.Var(z);
        });

        std::vector<TFNode> dW = _::map<Matrix, TFNode>(Parameters, [&CE](auto x) {
            Matrix z = Matrix::Zero(x.rows(), x.cols());
            return CE.Var(z);
        });

        return optimize(CE, X, Y, Parameters, opt_params, getGradient, getLoss, [&CE, rho, epsilon, &EG, &dW](TFNode P, TFNode G, int i) {
            return CE.ApplyAdadelta(P, EG[i], dW[i], 1.0, rho, epsilon, G);
        });
    };
}}
