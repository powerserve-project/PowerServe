// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "core/tensor.hpp"
#include "graph/op_params.hpp"
#include "graph/op_type.hpp"

#include <string>
#include <vector>

namespace powerserve {

enum class NodeType {
    TENSOR,
    OPERATOR,
    TENSOR_VIEW,
};

struct Graph;
struct TensorNode;
struct TensorViewNode;
struct OpNode;

struct Node {

public:
    NodeType type;
    std::string name = "";
    std::vector<Node *> prev;
    std::vector<Node *> next;

public:
    virtual ~Node() = default;

public:
    void connect(Node &other) {
        connect(&other);
    }

    void connect(Node *other) {
        next.push_back(other);
        other->prev.push_back(this);
    }

    void set_name(const std::string &name) {
        this->name = name;
    }

    auto tensor() -> Tensor *;
    auto op() -> OpNode *;
    auto tensor_view() -> TensorViewNode *;

protected:
    Node(NodeType type) : type(type) {}
};

struct TensorNode : Tensor, Node {
private:
    friend struct Graph;

protected:
    TensorNode(const Tensor &tensor) : Tensor(tensor), Node(NodeType::TENSOR) {}

    TensorNode(DataType dtype, const Shape &shape) : Tensor(dtype, shape), Node(NodeType::TENSOR) {}

public:
    virtual ~TensorNode() override = default;

public:
    auto prev_op() const -> OpNode * {
        POWERSERVE_ASSERT(prev.size() == 1);
        return prev[0]->op();
    }
};

struct TensorViewNode : TensorNode {
private:
    friend struct Graph;

public:
    Tensor *parent;

private:
    TensorViewNode(const Tensor &tensor, Shape shape) : TensorNode(tensor) {
        type   = NodeType::TENSOR_VIEW;
        parent = const_cast<Tensor *>(&tensor);
        POWERSERVE_ASSERT(parent->n_elements() == n_elements());
        m_shape = shape;
        m_data  = nullptr;
    }

public:
    ~TensorViewNode() override = default;
};

struct OpNode : Node {
public:
    OpType op;
    std::unique_ptr<OpParams> params;

private:
    OpNode(OpType op) : Node(NodeType::OPERATOR), op(op) {}

public:
    ~OpNode() override = default;

private:
    friend struct Graph;

public:
    void set_inputs(const std::vector<TensorNode *> &tensors) {
        for (auto tensor : tensors) {
            tensor->connect(this);
        }
    }

    void set_outputs(const std::vector<TensorNode *> &tensors) {
        for (auto tensor : tensors) {
            connect(tensor);
        }
    }

    template <typename T>
    void set_params(const T &params) {
        this->params.reset(new OpParamWrapper<T>(params));
    }

    template <typename T>
    const auto &get_params() const {
        return dynamic_cast<OpParamWrapper<T> *>(params.get())->value;
    }

    size_t n_outputs() const {
        return next.size();
    }

    auto output() const -> Tensor * {
        POWERSERVE_ASSERT(n_outputs() == 1);
        return next[0]->tensor();
    }
};

} // namespace powerserve
