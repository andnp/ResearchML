#pragma once

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tuple>

#include "util/cdash.hpp"
#include "ComputeEngine/matrix.hpp"

/* Thin wrapper over tensorflow operations
** Provides a few higher level methods for a more comprehensive API
** Attempts to convert every operation output to an Output
**/

// set tensorflow logging level. 3 disables all logging

namespace GPUCompute {
    void checkStatus(const Status &status);

class ComputeEngine {
    // Scope root = (Scope::NewRootScope()).WithDevice("/gpu:0");
    Scope root = Scope::NewRootScope();
    ClientSession *session;
    std::map<std::string, std::shared_ptr<Scope>> sub_scopes;
    std::vector<std::function<void()>> to_init = {};
    std::vector<
        std::tuple<
            tensorflow::Output,
            std::function<void(Matrix&)>>> captures = {};

    int hasInitialized = 0;
    int hasScope(std::string scope);
    Scope getSubScope(std::string scope);
    SessionOptions options = SessionOptions();

public:
    static bool FORCE_CPU;
    ComputeEngine();
    Matrix getMatrixFromTensor(tensorflow::Tensor &t);
    Tensor getTensorFromMatrix(const Matrix &M1);
    Scope getScope();
    ClientSession *getSession();
    TFNode Const(const Input::Initializer &data);
    TFNode Const(const Matrix &M);
    TFNode Assign(Input ref, Input val);
    TFNode Var(PartialTensorShape shape, const Input &I);
    TFNode Var(const Matrix &M);
    TFNode MatMul(Input a, Input b, const tensorflow::ops::MatMul::Attrs &attrs);
    TFNode MatMul(Input a, Input b);
    TFNode Sigmoid(Input a);
    TFNode Div(Input a, Input b);
    TFNode Add(Input a, Input b);
    TFNode AssignAdd(Input a, Input b);
    TFNode AssignSub(Input a, Input b);
    TFNode Concat(tensorflow::InputList vals, Input axis);
    TFNode Fill(Input dims, Input value);
    tensorflow::ops::SplitV SplitV(Input value, Input size_splits, Input axis, int64 num_split);
    TFNode Sub(Input a, Input b);
    TFNode Multiply(Input a, Input b);
    TFNode Sum(Input A, Input Axis);
    TFNode Conj(Input A);
    TFNode Sqrt(Input A);
    TFNode RandomShuffle(Input a, int seed);
    TFNode Identity(Input a);
    TFNode Transpose(Input a);
    TFNode SquaredDifference(Input a, Input b);
    TFNode OnesLike(Input a);
    TFNode ZerosLike(Input a);
    TFNode LikeX(Input a, Input x);
    TFNode Log(Input a);
    TFNode Min(Input a, Input b);
    TFNode Sign(Input a);
    TFNode Max(Input a, Input b);
    TFNode MaxLog(Input a);
    TFNode MatrixSum(Input a);
    TFNode Threshold(Input a, Input thresh);
    TFNode Print(Input a, InputList data, const tensorflow::ops::Print::Attrs &attrs);
    TFNode ApplyGradientDescent(Input w, Input alpha, Input grad);
    TFNode ApplyAdadelta(Input w, Input EG, Input dW, Input lr, Input rho, Input epsilon, Input grad);

    // Higher level interface
    TFNode InputVariable();
    tensorflow::OutputList InputVariables(int n);
    void InitializeVariables();
    void CaptureValues(tensorflow::Output a, std::function<void(Matrix&)> callback);
    TFNode Copy(Input T);
    std::vector<Tensor> run(const ClientSession::FeedType &inputs, const std::vector<TFNode> outputs);
    std::vector<Matrix> run(const tensorflow::OutputList &inputs, const std::vector<Matrix> &inits, const tensorflow::OutputList outputs);
    };
};  // namespace GPUCompute

