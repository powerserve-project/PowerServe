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

#include "ggml.h"
#include "model/model.hpp"
#include "sampler/sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace powerserve {

struct InternVL : Model {
public:
    // ggml need those context
    ggml_context *ggml_ctx;
    gguf_context *gguf_ctx;
    bool lazy_load;
    std::vector<std::pair<int, size_t>> img_infos;
    std::vector<std::vector<float>> pixel_values_list;
    static const int IMG_START = 151646;

public:
    explicit InternVL(const std::string &filename, const std::shared_ptr<ModelConfig> &config);
    ~InternVL() override;

public:
    // void generate(Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps) override;
    auto decode(Sampler &sampler, const std::vector<Token> tokens, const std::vector<int> pos, bool lm_head)
        -> std::vector<Token> override;
    auto generate(const Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps, size_t batch_size)
        -> std::shared_ptr<TokenIterator> override;
    auto preprocess(const std::vector<Path> &img_paths, const std::string &prompt) -> std::string;
    auto forward(
        const std::vector<int> &tokens,
        const std::vector<int> &pos,
        const CausalAttentionMask &mask,
        bool lm_head = true
    ) -> LogitsVector override;
};

} // namespace powerserve
