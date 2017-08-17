#pragma once

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

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
    ClientSession *session = new ClientSession(root);
    std::map<std::string, std::shared_ptr<Scope>> sub_scopes;
    std::vector<std::function<void()>> to_init = {};
    int hasInitialized = 0;
    int hasScope(std::string scope);
    Scope getSubScope(std::string scope);

public:
    Matrix getMatrixFromTensor(tensorflow::Tensor &t);
    Tensor getTensorFromMatrix(const Matrix &M1);
    Scope getScope();
    ClientSession *getSession();
    tensorflow::Output Const(const Input::Initializer &data);
    tensorflow::Output Const(const Matrix &M);
    tensorflow::Output Assign(Input ref, Input val);
    tensorflow::Output Var(PartialTensorShape shape, const Input &I);
    tensorflow::Output Var(const Matrix &M);
    tensorflow::Output MatMul(Input a, Input b, const tensorflow::ops::MatMul::Attrs &attrs);
    tensorflow::Output MatMul(Input a, Input b);
    tensorflow::Output Sigmoid(Input a);
    tensorflow::Output Div(Input a, Input b);
    tensorflow::Output Add(Input a, Input b);
    tensorflow::Output AssignAdd(Input a, Input b);
    tensorflow::Output AssignSub(Input a, Input b);
    tensorflow::Output Concat(tensorflow::InputList vals, Input axis);
    tensorflow::Output Fill(Input dims, Input value);
    tensorflow::ops::SplitV SplitV(Input value, Input size_splits, Input axis, int64 num_split);
    tensorflow::Output Sub(Input a, Input b);
    tensorflow::Output Multiply(Input a, Input b);
    tensorflow::Output Sum(Input A, Input Axis);
    tensorflow::Output Conj(Input A);
    tensorflow::Output Sqrt(Input A);

    // Higher level interface
    tensorflow::Output InputVariable();
    tensorflow::OutputList InputVariables(int n);
    void InitializeVariables();
    tensorflow::Output Copy(Input T);
    std::vector<Tensor> run(const ClientSession::FeedType &inputs, const std::vector<tensorflow::Output> outputs);
    std::vector<Matrix> run(const tensorflow::OutputList &inputs, const std::vector<Matrix> &inits, const tensorflow::OutputList outputs);
    };
};  // namespace GPUCompute

