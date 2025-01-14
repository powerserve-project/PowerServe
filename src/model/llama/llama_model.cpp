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

#include "llama_model.hpp"

#include "backend/cpu_buffer.hpp"
#include "core/logger.hpp"
#include "executor/executor.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "model/llama/llama_weight.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace powerserve {

LlamaModel::LlamaModel(const std::string &filename, const std::shared_ptr<ModelConfig> &config) : Model(filename) {
    {
        gguf_init_params params = {.no_alloc = false, .ctx = &ggml_ctx};
        gguf_ctx                = gguf_init_from_file(filename.c_str(), params);
        POWERSERVE_ASSERT(gguf_ctx != nullptr);
        POWERSERVE_ASSERT(ggml_ctx != nullptr);
    }
    m_config  = config;
    lazy_load = ggml_get_tensor(ggml_ctx, "output_norm.weight") == nullptr ? true : false;
    m_weights = std::make_shared<LlamaWeight>(ggml_ctx, m_config->llm.n_layers, lazy_load);
    if (lazy_load) {
        POWERSERVE_LOG_WARN("only the embedding table was loaded");
    }
    m_ffn = std::make_shared<FFN>(m_config->llm, m_weights);
}

LlamaModel::~LlamaModel() {
    gguf_free(gguf_ctx);
}

auto LlamaModel::forward(
    const std::vector<int> &tokens, const std::vector<int> &pos, const CausalAttentionMask &mask, bool lm_head
) -> LogitsVector {
    Graph g(m_config->model_id);
    // input embedding
    size_t batch_size  = tokens.size();
    auto embd_tb       = g.add_tensor(m_weights->token_embedding_table);
    auto x             = g.get_embedding(embd_tb, tokens);
    TensorNode *logits = nullptr;

    auto &llm_config = m_config->llm;

#if defined(POWERSERVE_WITH_QNN)
    if (m_platform->qnn_backend) {
        auto size            = llm_config.dim;
        bool use_qnn_lm_head = m_platform->qnn_backend->m_models[m_config->model_id]->m_config.lm_heads.size() > 0;
        if (use_qnn_lm_head) {
            size   = llm_config.vocab_size;
            logits = g.qnn_forward(x, pos, mask, size, lm_head);
        } else {
            x = g.qnn_forward(x, pos, mask, size, lm_head);
            if (lm_head) {
                auto rms_final_w    = g.add_tensor(m_weights->rms_final_weight);
                auto final_rms_norm = g.rms_norm(x, rms_final_w, llm_config.norm_eps);
                auto output_w       = g.add_tensor(m_weights->output_weight);
                logits              = g.mat_mul(output_w, final_rms_norm);
            }
        }
    } else
#endif
    {
        if (!lazy_load) {
            m_platform->ggml_backends[m_config->model_id]->reset_kv_batch_size(batch_size);
            for (size_t L = 0; L < llm_config.n_layers; L++) {
                auto [k_cache, v_cache] = m_platform->ggml_backends[m_config->model_id]->m_kv->get_cache(L);
                auto att_o = m_attn->build(g, x, L, g.add_tensor(k_cache), g.add_tensor(v_cache), pos, mask);
                auto ffn_o = m_ffn->build(g, att_o, L);
                x          = ffn_o;
            }
            // TODO: cpu and qnn reuse
            if (lm_head) {
                auto rms_final_w    = g.add_tensor(m_weights->rms_final_weight);
                auto final_rms_norm = g.rms_norm(x, rms_final_w, llm_config.norm_eps);
                auto output_w       = g.add_tensor(m_weights->output_weight);
                logits              = g.mat_mul(output_w, final_rms_norm);
            }
        }
    }

    Executor executor(*m_platform, g);
    executor.allocate_buffers();

    executor.run();
#if defined(POWERSERVE_WITH_QNN)
    if (!m_platform->qnn_backend)
#endif
    {
        m_platform->ggml_backends[m_config->model_id]->m_kv->advance(batch_size);
    }

    if (!lm_head) {
        return LogitsVector();
    }

    return LogitsVector(logits->m_data, m_config->llm.vocab_size, batch_size);
}

auto LlamaModel::decode(Sampler &sampler, const std::vector<Token> tokens, const std::vector<int> pos, bool lm_head)
    -> std::vector<Token> {
    auto mask = CausalAttentionMask(tokens.size());
    auto ret  = forward(tokens, pos, mask, lm_head);
    std::vector<Token> toks;
    for (auto logits : ret.logits_vector) {
        auto probs = ProbArray(logits);
        sampler.apply(probs);
        auto next = probs.greedy_sample().token;
        sampler.accept(next);
        toks.push_back(next);
    }
    return toks;
}

auto LlamaModel::generate(
    const Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps, size_t batch_size
) -> std::shared_ptr<TokenIterator> {
    return std::make_shared<ModelTokenIterator>(*this, tokenizer, sampler, prompt, steps, batch_size);
}

} // namespace powerserve
