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

#include "speculative/token_tree.hpp"

#include "core/getenv.hpp"
#include "core/perfetto_trace.hpp"
#include "nlohmann/json.hpp"

#include <fstream>

namespace {

using powerserve::getenv;

static auto dump_file_path = getenv<std::string>("dump_file", "");

} // namespace

namespace powerserve {

TokenTree::TokenTree(const SpeculativeConfig &config) : config(config) {
    draft_sampler.append<TopKSampler>(config.draft_sampler.top_k);
    draft_sampler.append<TemperatureSampler>(config.draft_sampler.temperature);
    draft_sampler.append<SoftmaxSampler>();
}

TokenTree::~TokenTree() {
    if (!dump_file_path.empty()) {
        nlohmann::json json = {
            {"n_draft_times", stat.n_draft_times},
            {"n_draft_tokens", stat.n_draft_tokens},
            {"n_accepted_tokens", stat.n_accepted_tokens},
            {"n_iterations", stat.n_iterations},
            {"n_generated_tokens", stat.n_generated_tokens},
        };

        std::ofstream dump_file(dump_file_path);
        POWERSERVE_ASSERT(dump_file.is_open());
        dump_file << json.dump() << std::endl;
    }
}

auto TokenTree::tokens() const -> std::vector<Token> {
    const size_t batch_size = nodes.size();
    std::vector<int> vec(batch_size);

    for (size_t u = 0; u < batch_size; u++) {
        vec[u] = nodes[u].token;
    }

    return vec;
}

auto TokenTree::positions() const -> std::vector<int> {
    const size_t batch_size = nodes.size();
    std::vector<int> vec(batch_size);

    for (size_t u = 0; u < batch_size; u++) {
        vec[u] = nodes[u].position;
    }

    return vec;
}

auto TokenTree::attention_mask() const -> std::vector<std::vector<bool>> {
    const size_t batch_size = nodes.size();
    std::vector<std::vector<bool>> mask(batch_size);

    for (size_t u = 0; u < nodes.size(); u++) {
        mask[u].resize(batch_size);

        for (int x = u; x != Node::no_parent; x = nodes[x].parent) {
            mask[u][x] = true;
        }
    }

    return mask;
}

void TokenTree::draft(const ModelPtr &draft_model, const Tokenizer &tokenizer, size_t batch_size, Token root_token) {
    counter = PerfettoTrace::counter();

    reset(batch_size);

    main_heap.push(Candidate{
        .token           = root_token,
        .parent          = Node::no_parent,
        .current_prob    = 1.0f,
        .cumulative_prob = 1.0f,
    });

    int last_parent       = Node::no_parent;
    size_t n_nodes        = 0;
    size_t n_saved_tokens = 0;
    CausalAttentionMask mask(1);
    while (n_nodes < batch_size) {
        bool is_leaf = main_heap.empty();
        auto &heap   = is_leaf ? leaf_heap : main_heap;

        if (heap.empty()) {
            break;
        }

        auto [token, parent, current_prob, cumulative_prob] = heap.top();
        heap.pop();

        int u             = n_nodes++;
        auto &node        = nodes[u];
        node.token        = token;
        node.current_prob = current_prob;

        if (parent == Node::no_parent) {
            node.position = draft_model->kv_cache->position;
        } else {
            node.position = nodes[parent].position + 1;
            node.parent   = parent;
            node.depth    = nodes[parent].depth + 1;
            nodes[parent].children.push_back(u);
        }

        // Early terminate
        if (is_leaf || tokenizer.should_stop(node.token) ||
            n_nodes + (config.token_tree.early_stop ? main_heap.size() / 2 : 0) >= batch_size ||
            cumulative_prob < config.token_tree.min_prob) {
            continue;
        }

        if (last_parent != Node::no_parent) {
            switch_parent(draft_model, last_parent, parent);
        }

        node.cache_index = draft_model->kv_cache->position;

        PerfettoTrace::begin("draft_model_forward");
        auto ret = draft_model->forward({node.token}, {node.position}, CausalAttentionMask(1));
        PerfettoTrace::end();

        n_saved_tokens++;
        last_parent = u;
        ProbArray probs(ret.logits_vector[0]);
        draft_sampler.apply(probs);
        POWERSERVE_ASSERT(probs.m_is_normalized);
        POWERSERVE_ASSERT(probs.m_is_sorted);

        float min_prob = probs[0].prob * config.draft_sampler.p_base;
        for (size_t i = 0; auto &item : probs) {
            bool leaf_only = (i >= config.token_tree.max_fan_out || item.prob < min_prob);
            i++;

            auto &heap = leaf_only ? leaf_heap : main_heap;
            heap.push(Candidate{
                .token           = item.token,
                .parent          = u,
                .current_prob    = item.prob,
                .cumulative_prob = cumulative_prob * item.prob,
            });
        }
    }

    stat.n_draft_times += n_saved_tokens;
    stat.n_draft_tokens += n_nodes - 1; // Exclude root token
    draft_model->kv_cache->rollback_tokens(n_saved_tokens);
}

void TokenTree::verify(
    const ModelPtr &target_model,
    const ModelPtr &draft_model,
    Sampler &sampler,
    std::vector<std::span<const float>> &logits,
    const EnqueueTokenFn &enqueue_token
) {
    POWERSERVE_ASSERT(target_model->kv_cache->position == draft_model->kv_cache->position);
    stat.n_iterations += 1;

    int u                     = 0;
    size_t n_generated_tokens = 0;
    while (true) {
        auto &node    = nodes[u];
        node.accepted = true;

        POWERSERVE_ASSERT((int)draft_model->kv_cache->position == node.position);
        POWERSERVE_ASSERT((int)target_model->kv_cache->position == node.position);

        target_model->kv_cache->copy(node.position, u);
        target_model->kv_cache->advance_tokens(1);

        if (node.cache_index == Node::not_in_cache) {
            // To catch up with target model
            PerfettoTrace::begin("draft_model_catch_up");
            draft_model->forward({node.token}, {node.position}, CausalAttentionMask(1), false);
            PerfettoTrace::end();
        } else {
            POWERSERVE_ASSERT(node.cache_index >= node.position);
            draft_model->kv_cache->move(node.position, node.cache_index);
            draft_model->kv_cache->advance_tokens(1);
        }

        ProbArray probs(logits[u]);
        sampler.apply(probs);
        auto next_token = probs.greedy_sample().token;
        enqueue_token(next_token);
        n_generated_tokens += 1;

        auto it = std::find_if(node.children.begin(), node.children.end(), [next_token, this](int v) {
            return nodes[v].token == next_token;
        });

        if (it == node.children.end()) {
            break;
        } else {
            u = *it;
            stat.n_accepted_tokens += 1;
        }
    }

    stat.n_generated_tokens += n_generated_tokens;
    counter.set_value("#generated-token", n_generated_tokens);
}

void TokenTree::print_tree(const Tokenizer &tokenizer, int u) {
    auto &node = nodes[u];
    auto piece = tokenizer.to_string(node.token);
    fmt::println("{:{}}[{}] {:?} {:.2f}", "", node.depth, (node.accepted ? "*" : " "), piece, node.current_prob);

    for (int v : node.children) {
        print_tree(tokenizer, v);
    }
}

void TokenTree::print_stat() {
    fmt::println("Speculative token tree statistics:");
    fmt::println("- {} iterations, {} generated tokens", stat.n_iterations, stat.n_generated_tokens);
    fmt::println("- {:.3f} tokens/iteration", 1.0 * stat.n_generated_tokens / stat.n_iterations);
    fmt::println("- {:.3f} draft-forwards/iteration", 1.0 * stat.n_draft_times / stat.n_iterations);
    fmt::println("- Accept ratio: {:.3f}%", 100.0 * stat.n_accepted_tokens / stat.n_draft_tokens);
    fmt::println("- Draft effective ratio: {:.3f}%", 100.0 * stat.n_accepted_tokens / stat.n_draft_times);
}

void TokenTree::Node::reset() {
    parent       = no_parent;
    depth        = 0;
    token        = 0;
    position     = 0;
    cache_index  = not_in_cache;
    current_prob = 1.0f;
    accepted     = false;
    children.clear();
}

void TokenTree::reset(size_t batch_size) {
    nodes.resize(batch_size);
    for (auto &node : nodes) {
        node.reset();
    }

    while (!main_heap.empty()) {
        main_heap.pop();
    }
    while (!leaf_heap.empty()) {
        leaf_heap.pop();
    }
}

int TokenTree::lca(int u, int v) const {
    if (nodes[u].depth < nodes[v].depth) {
        std::swap(u, v);
    }

    while (nodes[u].depth > nodes[v].depth) {
        u = nodes[u].parent;
    }

    while (u != v) {
        u = nodes[u].parent;
        v = nodes[v].parent;
    }

    return u;
}

void TokenTree::switch_parent(const ModelPtr &draft_model, int old_parent, int new_parent) {
    if (old_parent == new_parent) {
        return;
    }

    int p = lca(old_parent, new_parent);

    while (old_parent != p) {
        POWERSERVE_ASSERT(nodes[old_parent].cache_index != Node::not_in_cache);
        draft_model->kv_cache->mask(nodes[old_parent].cache_index);
        old_parent = nodes[old_parent].parent;
    }

    while (new_parent != p) {
        POWERSERVE_ASSERT(nodes[new_parent].cache_index != Node::not_in_cache);
        draft_model->kv_cache->unmask(nodes[new_parent].cache_index);
        new_parent = nodes[new_parent].parent;
    }
}

} // namespace powerserve
