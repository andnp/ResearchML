#include "loss.hpp"
#include "ComputeEngine/ComputeEngine.hpp"

namespace GPUCompute {
namespace Loss {
    TFNode leastSquaresLoss(ComputeEngine &CE, Input Yhat, Input Y, int samples) {
        Numeric_t t = static_cast<Numeric_t>(samples);
        auto d = CE.SquaredDifference(Y, Yhat);
        auto s = CE.MatrixSum(d);
        return CE.Div(s, t);
    }

    TFNode leastSquaresLossGradient(ComputeEngine &CE, Input Yhat, Input Y, int samples) {
        Numeric_t t = static_cast<Numeric_t>(samples);
        Numeric_t two = 2.0;
        return CE.Div(CE.Multiply(two, CE.Sub(Yhat, Y)), t);
    }

    TFNode crossEntropyLoss(ComputeEngine &CE, TFNode P, TFNode Y, int samples) {
        Numeric_t t = static_cast<Numeric_t>(samples);
        // Y * log(P)
        auto a = CE.Multiply(Y, CE.MaxLog(P));
        // (1 - Y) * log(1 - P)
        Numeric_t one = 1.0;
        auto b = CE.Multiply(CE.Sub(one, Y), CE.MaxLog(CE.Sub(one, P)));
        auto c = CE.Add(a, b);
        auto S = CE.MatrixSum(c);
        return CE.Multiply(CE.Div(-one, t), S);
    }

    TFNode gradientCrossEntropy(ComputeEngine &CE, TFNode Yhat, TFNode Y, int samples) {
        Numeric_t t = static_cast<Numeric_t>(samples);
        return CE.Div(CE.Sub(Yhat, Y), t);
    };

    TFNode sigmoidGradient(ComputeEngine &CE, TFNode X) {
        Numeric_t one = 1.0;
        return CE.Multiply(X, CE.Sub(one, X));
    }

    TFNode l2Norm(ComputeEngine &CE, TFNode X, int samples) {
        return CE.Div(CE.MatrixSum(CE.Multiply(X, X)), static_cast<Numeric_t>(samples));
    }
}}
