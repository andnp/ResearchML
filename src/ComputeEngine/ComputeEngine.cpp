#include "ComputeEngine.hpp"

namespace GPUCompute {
    void checkStatus(const Status &status) {
        if (!status.ok()) {
            std::cout << status.ToString() << std::endl;
            exit(1);
        }
    }
    /*****************
    * Private Methods
    ******************/
    int ComputeEngine::hasScope(std::string scope) {
        if (sub_scopes.count(scope)) return 1;
        else return 0;
    };

    Scope ComputeEngine::getSubScope(std::string scope) {
        const int exists = hasScope(scope);
        if (exists == 0) sub_scopes[scope] = std::make_shared<Scope>(root.NewSubScope(scope));
        return *sub_scopes[scope];
    };

    /****************
    * Public Methods
    *****************/

    Matrix ComputeEngine::getMatrixFromTensor(tensorflow::Tensor &t) {
        // If is scalar, then has 0 dimensions
        if (t.dims() == 0) {
            auto m = Matrix(1, 1);
            m(0, 0) = t.flat<Numeric_t>().data()[0];
            return m;
        } else if (t.dims() == 1) {
            // For now always return a column vector
            auto m = Eigen::Map<Eigen::Matrix<
                Numeric_t,
                Eigen::Dynamic,
                Eigen::Dynamic,
                Eigen::RowMajor>>(
                t.flat<Numeric_t>().data(),
                t.dim_size(0),
                1);
            return m;
        } else {
            auto m = Eigen::Map<Eigen::Matrix<
                Numeric_t,       /* scalar element type */
                Eigen::Dynamic, /* num_rows is a run-time value */
                Eigen::Dynamic, /* num_cols is a run-time value */
                Eigen::RowMajor /* tensorflow::Tensor is always row-major */>>(
                t.flat<Numeric_t>().data(), /* ptr to data */
                t.dim_size(0),             /* num_rows */
                t.dim_size(1) /* num_cols */);
            return m;
        }
    };

    Tensor ComputeEngine::getTensorFromMatrix(const Matrix &M1) {
        int r = M1.rows();
        int c = M1.cols();
        Tensor T(Tensor_t,
                    TensorShape({M1.rows(), M1.cols()}));
        auto dat = T.flat<Numeric_t>().data();
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                dat[j + (i * c)] = M1(i, j);
            }
        }
        return T;
    };

    Scope ComputeEngine::getScope() { return root; };
    ClientSession* ComputeEngine::getSession() { return session; };
    tensorflow::Output ComputeEngine::InputVariable() { return Placeholder(root, Tensor_t); };

    tensorflow::OutputList ComputeEngine::InputVariables(int n) { return _::times<tensorflow::Output>(n, [this](int i) { return InputVariable();}); }

    tensorflow::Output ComputeEngine::Const(const Input::Initializer &data) {
        return tensorflow::ops::Const(root, data);
    };

    tensorflow::Output ComputeEngine::Const(const Matrix &M) {
        return ComputeEngine::Const(getTensorFromMatrix(M));
    };

    tensorflow::Output ComputeEngine::Assign(Input ref, Input val) {
        return tensorflow::ops::Assign(root, ref, val);
    };

    tensorflow::Output ComputeEngine::Var(PartialTensorShape shape, const Input &I) {
        Scope init = getSubScope("init");
        auto var = Variable(root, shape, Tensor_t);
        auto assignment_op = tensorflow::ops::Assign(init, var, I);
        to_init.push_back([assignment_op, this]() {
            TF_CHECK_OK(this->session->Run({}, {}, {Operation(assignment_op.node())}, nullptr));
        });
        return var;
    };

    tensorflow::Output ComputeEngine::Var(const Matrix &M) {
        auto T = getTensorFromMatrix(M);
        return Var({M.rows(), M.cols()}, T);
    };

    void ComputeEngine::InitializeVariables() {
        if (hasInitialized == 1) return;
        _::forEach(to_init, [](auto f) {f();});
        hasInitialized = 1;
    };

    tensorflow::Output ComputeEngine::MatMul(Input a, Input b, const tensorflow::ops::MatMul::Attrs &attrs) {
        return tensorflow::ops::MatMul(root, a, b, attrs);
    };

    tensorflow::Output ComputeEngine::MatMul(Input a, Input b) {
        return tensorflow::ops::MatMul(root, a, b);
    };

    tensorflow::Output ComputeEngine::Sigmoid(Input a) {
        return tensorflow::ops::Sigmoid(root, a);
    };

    tensorflow::Output ComputeEngine::Div(Input a, Input b) {
        return tensorflow::ops::Div(root, a, b);
    };

    tensorflow::Output ComputeEngine::Add(Input a, Input b) {
        return tensorflow::ops::Add(root, a, b);
    };

    tensorflow::Output ComputeEngine::AssignAdd(Input a, Input b) {
        return tensorflow::ops::AssignAdd(root, a, b);
    };

    tensorflow::Output ComputeEngine::AssignSub(Input a, Input b) {
        return tensorflow::ops::AssignSub(root, a, b);
    };

    tensorflow::Output ComputeEngine::Concat(tensorflow::InputList vals, Input axis) {
        return tensorflow::ops::Concat(root, vals, axis);
    };

    tensorflow::Output ComputeEngine::Fill(Input dims, Input value) {
        return tensorflow::ops::Fill(root, dims, value);
    };

    tensorflow::ops::SplitV ComputeEngine::SplitV(Input value, Input size_splits, Input axis, int64 num_split) {
        return tensorflow::ops::SplitV(root, value, size_splits, axis, num_split);
    };

    tensorflow::Output ComputeEngine::Sub(Input a, Input b) {
        return tensorflow::ops::Sub(root, a, b);
    };

    tensorflow::Output ComputeEngine::Multiply(Input a, Input b) {
        return tensorflow::ops::Multiply(root, a, b);
    };

    tensorflow::Output ComputeEngine::Sum(Input A, Input Axis) {
        return tensorflow::ops::Sum(root, A, Axis);
    };

    tensorflow::Output ComputeEngine::Sqrt(Input A) {
        return tensorflow::ops::Sqrt(root, A);
    };

    tensorflow::Output ComputeEngine::Conj(Input A) {
        return tensorflow::ops::Conj(root, A);
    };

    tensorflow::Output ComputeEngine::Copy(Input T) {
        auto Z = ZerosLike(root, T);
        return Sub(T, Z);
    };

    TFNode ComputeEngine::RandomShuffle(Input a, int seed) {
        return tensorflow::ops::RandomShuffle(root, a, tensorflow::ops::RandomShuffle::Seed(seed));
    };

    TFNode ComputeEngine::Transpose(Input a) {
        return tensorflow::ops::Transpose(root, a, {1, 0});
    };

    TFNode ComputeEngine::Identity(Input a) {
        return tensorflow::ops::Identity(root, a);
    };

    TFNode ComputeEngine::SquaredDifference(Input a, Input b) {
        return tensorflow::ops::SquaredDifference(root, a, b);
    };

    TFNode ComputeEngine::Log(Input a) {
        return tensorflow::ops::Log(root, a);
    };

    TFNode ComputeEngine::Max(Input a, Input b) {
        return tensorflow::ops::Maximum(root, a, b);
    };

    TFNode ComputeEngine::MaxLog(Input a) {
        return Max(Log(a), -1e8);
    };

    TFNode ComputeEngine::MatrixSum(Input a) {
        return Sum(Sum(a, 1), 0);
    };

    TFNode ComputeEngine::Print(Input a, InputList data, const tensorflow::ops::Print::Attrs &attrs) {
        return tensorflow::ops::Print(root, a, data, attrs);
    };

    TFNode ComputeEngine::ApplyGradientDescent(Input w, Input alpha, Input grad) {
        return tensorflow::ops::ApplyGradientDescent(root, w, alpha, grad);
    };


    std::vector<Tensor> ComputeEngine::run(const ClientSession::FeedType& inputs, const std::vector<tensorflow::Output> outputs) {
        InitializeVariables();
        std::vector<Tensor> outs;
        TF_CHECK_OK(this->session->Run(inputs, outputs, &outs));
        return outs;
    };

    std::vector<Matrix> ComputeEngine::run(const tensorflow::OutputList &inputs, const std::vector<Matrix> &inits, const tensorflow::OutputList outputs) {
        std::vector<Tensor> tensors = {};
        for (int i = 0; i < inits.size(); ++i) {
            tensors.push_back(getTensorFromMatrix(inits[i]));
        }

        ClientSession::FeedType feed = {};
        for (int i = 0; i < inits.size(); ++i) {
            feed.insert(std::make_pair(inputs[i], tensors[i]));
        }

        auto outs = run(feed, outputs);
        std::vector<Matrix> out_mats = {};
        for (int i = 0; i < outs.size(); ++i) {
            out_mats.push_back(getMatrixFromTensor(outs[i]));
        }

        return out_mats;
    }
};
