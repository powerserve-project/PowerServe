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

#include "graph/node.hpp"

namespace powerserve {

auto Node::tensor() -> Tensor * {
    return dynamic_cast<Tensor *>(this);
}

auto Node::tensor_view() -> TensorViewNode * {
    return dynamic_cast<TensorViewNode *>(this);
}

auto Node::op() -> OpNode * {
    return dynamic_cast<OpNode *>(this);
}

} // namespace powerserve
