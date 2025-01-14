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

#include "backend/platform.hpp"
#include "model/module/attention.hpp"
#include "model/module/ffn.hpp"
#include "sampler/sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <string>

namespace powerserve {

struct LogitsVector {
    BufferPtr buffer;
    std::vector<std::span<const float>> logits_vector;

    LogitsVector() = default;

    LogitsVector(BufferPtr buffer, size_t vocab_size, size_t batch_size) : buffer(buffer) {
        float *logits = static_cast<float *>(dynamic_cast<CPUBuffer &>(*buffer).m_data);
        for (size_t i = 0; i < batch_size; i++) {
            logits_vector.push_back(std::span<const float>(logits, logits + vocab_size));
            logits += vocab_size;
        }
    }
};

struct TokenIterator {
    size_t n_rest              = 0;
    std::string m_prompt       = "";
    size_t m_batch_size        = 1;
    std::deque<Token> m_tokens = {};

    const Tokenizer &m_tokenizer;
    Sampler &m_sampler;

    TokenIterator(
        size_t n_rest,
        const std::string &m_prompt,
        size_t m_batch_size,
        const Tokenizer &m_tokenizer,
        Sampler &m_sampler
    ) :
        n_rest(n_rest),
        m_prompt(m_prompt),
        m_batch_size(m_batch_size),
        m_tokenizer(m_tokenizer),
        m_sampler(m_sampler) {}

    virtual ~TokenIterator() = default;

    virtual auto next() -> Token {
        auto next = m_tokens.front();
        decode();
        return next;
    }

    virtual auto end() -> bool {
        return n_rest <= 0;
    }

    virtual void decode() = 0;
};

struct Model {
public:
    std::string m_filename;
    std::shared_ptr<ModelConfig> m_config;
    std::shared_ptr<Weight> m_weights;
    std::shared_ptr<Attention> m_attn;
    std::shared_ptr<FFN> m_ffn;
    std::shared_ptr<Platform> m_platform;
    KVCacheInterface *kv_cache = nullptr;

public:
    Model(const std::string &filename) :
        m_filename(filename),
        m_config(nullptr),
        m_weights(nullptr),
        m_attn(nullptr),
        m_ffn(nullptr),
        m_platform(nullptr) {}

    virtual ~Model() = default;

    virtual auto forward(
        const std::vector<int> &tokens,
        const std::vector<int> &pos,
        const CausalAttentionMask &mask,
        bool lm_head = true
    ) -> LogitsVector = 0;

public:
    virtual auto decode(Sampler &sampler, const std::vector<Token> tokens, const std::vector<int> pos, bool lm_head)
        -> std::vector<Token> = 0;
    virtual auto generate(
        const Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps, size_t batch_size
    ) -> std::shared_ptr<TokenIterator> = 0;
};

using ModelPtr = std::shared_ptr<Model>;

struct ModelTokenIterator : TokenIterator {
public:
    Model &m_model;

public:
    ModelTokenIterator(
        Model &model,
        const Tokenizer &tokenizer,
        Sampler &sampler,
        const std::string &prompt,
        int steps,
        size_t batch_size
    ) :
        TokenIterator(steps, prompt, batch_size, tokenizer, sampler),
        m_model(model) {
        if (steps <= 0) {
            return;
        }

        auto prompt_tokens   = m_tokenizer.tokenize(m_prompt, m_tokenizer.m_vocab.tokenizer_add_bos);
        auto n_prompt_tokens = prompt_tokens.size();
        size_t n_prefilled   = 0;
        size_t position      = 0;

        auto &m_platform = m_model.m_platform;
        auto &model_id   = m_model.m_config->model_id;
        m_platform->reset_kv_position(model_id);
        position = m_platform->get_kv_position(model_id);
        m_platform->ggml_backends[model_id]->setup_threadpool();
        // prefill
        while (n_prefilled < n_prompt_tokens - 1) {
            size_t bs = std::min(m_batch_size, n_prompt_tokens - n_prefilled - 1);
            std::vector<Token> tokens;
            std::copy(
                prompt_tokens.begin() + n_prefilled,
                prompt_tokens.begin() + n_prefilled + bs,
                std::back_inserter(tokens)
            );
            std::vector<int> pos(bs);
            std::iota(pos.begin(), pos.end(), position);
            m_model.decode(m_sampler, tokens, pos, false);
            position = m_platform->get_kv_position(model_id);
            n_prefilled += bs;
        }
        position = m_platform->get_kv_position(model_id);
        m_tokens.push_back(prompt_tokens.back());
    }

    ~ModelTokenIterator() {
        // TODO speculative's reset
        m_model.m_platform->ggml_backends[m_model.m_config->model_id]->reset_threadpool();
    }

    virtual void decode() override {
        if (n_rest > 0 && m_tokens.size() >= 1) {
            if (m_tokens.size() == 1) {
                auto &platform   = m_model.m_platform;
                auto current_pos = platform->get_kv_position(m_model.m_config->model_id);
                std::vector<int> pos(1, current_pos);
                std::vector<int> token(1, m_tokens.front());
                auto ret = m_model.decode(m_sampler, token, pos, true);
                std::copy(ret.begin(), ret.end(), std::back_inserter(m_tokens));
            }
            --n_rest;
            m_tokens.pop_front();
        }
    }
};

} // namespace powerserve
