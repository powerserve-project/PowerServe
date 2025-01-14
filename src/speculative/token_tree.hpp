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

#include "core/perfetto_trace.hpp"
#include "model/model.hpp"
#include "sampler/sampler_chain.hpp"
#include "speculative/speculative_config.hpp"

#include <functional>
#include <queue>

namespace powerserve {

struct TokenTree {
    using EnqueueTokenFn = std::function<void(Token token)>;

    TokenTree(const SpeculativeConfig &config);
    ~TokenTree();

    auto tokens() const -> std::vector<Token>;
    auto positions() const -> std::vector<int>;
    auto attention_mask() const -> std::vector<std::vector<bool>>;

    void draft(const ModelPtr &draft_model, const Tokenizer &tokenizer, size_t batch_size, Token root_token);
    void verify(
        const ModelPtr &target_model,
        const ModelPtr &draft_model,
        Sampler &sampler,
        std::vector<std::span<const float>> &logits,
        const EnqueueTokenFn &enqueue_token
    );

    // For debugging
    void print_tree(const Tokenizer &tokenizer, int u = 0);
    void print_stat();

private:
    SpeculativeConfig config;
    SamplerChain draft_sampler;

    struct Node {
        static constexpr int no_parent    = -1;
        static constexpr int not_in_cache = -1;

        int parent         = no_parent;
        int depth          = 0;
        Token token        = 0;
        int position       = 0;
        int cache_index    = not_in_cache;
        float current_prob = 1.0f;  // For debugging
        bool accepted      = false; // For debugging
        std::vector<int> children;

        void reset();
    };

    std::vector<Node> nodes;

    struct Candidate {
        Token token;
        int parent;
        float current_prob;
        float cumulative_prob;

        bool operator<(const Candidate &other) const {
            return cumulative_prob < other.cumulative_prob;
        }
    };

    std::priority_queue<Candidate> main_heap;
    std::priority_queue<Candidate> leaf_heap;

    struct {
        size_t n_draft_times      = 0;
        size_t n_draft_tokens     = 0;
        size_t n_accepted_tokens  = 0;
        size_t n_iterations       = 0;
        size_t n_generated_tokens = 0;
    } stat;

    PerfettoTrace::CounterTimePoint counter;

    void reset(size_t batch_size);

    // Lowest common ancestor
    int lca(int u, int v) const;

    // When generating a new token at a new parent, draft model's attention mask should be
    // updated to ignore tokens from other branches.
    void switch_parent(const ModelPtr &draft_model, int old_parent, int new_parent);
};

} // namespace powerserve
