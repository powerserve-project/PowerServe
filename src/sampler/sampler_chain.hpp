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

#include "core/config.hpp"
#include "sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <vector>

namespace powerserve {

struct SamplerChain final : Sampler {
    virtual ~SamplerChain() override = default;

    SamplerChain() = default;

    SamplerChain(const HyperParams::SamplerConfig &config, const Tokenizer &tokenizer) {
        build_from_config(config, tokenizer);
    }

    template <typename SamplerType, typename... Args>
    void append(Args &&...args) {
        m_samplers.emplace_back(std::make_unique<SamplerType>(std::forward<Args>(args)...));
    }

    void build_from_config(const HyperParams::SamplerConfig &config, const Tokenizer &tokenizer);

    void apply(ProbArray &probs) override;
    void accept(Token token) override;

private:
    std::vector<std::unique_ptr<Sampler>> m_samplers;
};

} // namespace powerserve
