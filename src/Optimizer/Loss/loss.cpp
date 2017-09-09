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
        return CE.Div(CE.Multiply(2.0, CE.Sub(Yhat, Y)), t);
    }

    TFNode crossEntropyLoss(ComputeEngine &CE, TFNode P, TFNode Y, int samples) {
        Numeric_t t = static_cast<Numeric_t>(samples);
        // Y * log(P)
        auto a = CE.Multiply(Y, CE.MaxLog(P));
        // (1 - Y) * log(1 - P)
        auto b = CE.Multiply(CE.Sub(1.0, Y), CE.MaxLog(CE.Sub(1.0, P)));
        auto c = CE.Add(a, b);
        auto S = CE.MatrixSum(c);
        return CE.Multiply(CE.Div(-1.0, t), S);
    }

    TFNode gradientCrossEntropy(ComputeEngine &CE, TFNode Yhat, TFNode Y, int samples) {
        Numeric_t t = static_cast<Numeric_t>(samples);
        return CE.Div(CE.Sub(Yhat, Y), t);
    };

    TFNode sigmoidGradient(ComputeEngine &CE, TFNode X) {
        return CE.Multiply(X, CE.Sub(1.0, X));
    }

    TFNode l2Norm(ComputeEngine &CE, TFNode X, int samples) {
        return CE.Div(CE.MatrixSum(CE.Multiply(W, W)), static_cast<Numeric_t>(samples));
    }
}}
