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

#include "sampler.hpp"

namespace powerserve {

void TemperatureSampler::apply(ProbArray &probs) {
    POWERSERVE_ASSERT(m_temperature > 0);

    if (m_temperature != 1) {
        for (auto &prob : probs.m_probs) {
            prob.prob /= m_temperature;
        }

        probs.m_is_normalized = false;
    }
}

void SoftmaxSampler::apply(ProbArray &probs) {
    probs.softmax();
}

void NormalizeSampler::apply(ProbArray &probs) {
    probs.normalize();
}

void TopKSampler::apply(ProbArray &probs) {
    POWERSERVE_ASSERT(m_topk > 0);
    auto k = std::min(m_topk, probs.m_probs.size());

    // Sort scores in descending order
    if (!probs.m_is_sorted) {
        std::partial_sort(
            probs.m_probs.begin(), probs.m_probs.begin() + k, probs.m_probs.end(), std::greater<ProbIndex>{}
        );
        probs.m_is_sorted = true;
    }

    if (k != probs.m_probs.size()) {
        probs.m_is_normalized = false;
    }

    probs.m_probs.resize(k);
}

void TopPSampler::apply(ProbArray &probs) {
    if (m_topp >= 1.0f) {
        return;
    }
    POWERSERVE_ASSERT(probs.m_is_normalized);
    POWERSERVE_ASSERT(probs.m_is_sorted);

    // Compute the cumulative probabilities
    float cum_sum   = 0.0f;
    size_t last_idx = probs.m_probs.size();

    for (size_t i = 0; i < probs.m_probs.size(); ++i) {
        cum_sum += probs.m_probs[i].prob;

        // Check if the running sum is at least p or if we have kept at least min_keep tokens
        // we set the last index to i+1 to indicate that the current iterate should be included in the set
        if (cum_sum >= m_topp && i + 1 >= m_min_keep) {
            last_idx = i + 1;
            break;
        }
    }

    if (last_idx != probs.m_probs.size()) {
        probs.m_is_normalized = false;
    }

    probs.m_probs.resize(last_idx);
}

void RepeatPenaltySampler::apply(ProbArray &probs) {
    if (m_ignore_eos) {
        // if ignore eos, set the logit of eos token to -INFINITY, so it will not be selected
        if (probs.m_probs.size() > (size_t)m_special_eos_id &&
            probs.m_probs[m_special_eos_id].token == m_special_eos_id) {
            probs.m_probs[m_special_eos_id].prob = -INFINITY;
        } else {
            // search and set the logit of eos token to -INFINITY
            for (size_t i = 0; i < probs.m_probs.size(); ++i) {
                if (probs.m_probs[i].token == m_special_eos_id) {
                    probs.m_probs[i].prob = -INFINITY;
                    break;
                }
            }
        }
    }

    if ((m_penalty_last_n == 0) || (m_penalty_repeat == 1.0f && m_penalty_freq == 0.0f && m_penalty_present == 0.0f)) {
        return;
    }

    bool nl_found  = false;
    size_t nl_idx  = 0;
    float nl_logit = -INFINITY;
    if (!m_penalize_nl) {
        // if not penalize nl, save its original logit value, so we can restore it later
        POWERSERVE_ASSERT(m_linefeed_id >= 0);

        // optimistically check if the candidates are not yet sorted/shuffled/truncated
        if (probs.m_probs.size() > (size_t)m_linefeed_id && probs.m_probs[m_linefeed_id].token == m_linefeed_id) {
            nl_found = true;
            nl_idx   = m_linefeed_id;
            nl_logit = probs.m_probs[m_linefeed_id].prob;
        } else {
            // else, search for the linefeed token
            for (size_t i = 0; i < probs.m_probs.size(); ++i) {
                if (probs.m_probs[i].token == m_linefeed_id) {
                    nl_found = true;
                    nl_idx   = i;
                    nl_logit = probs.m_probs[i].prob;
                    break;
                }
            }
        }
    }

    // Create a frequency map to count occurrences of each token in last_tokens
    // TODO: optimize this by maintaining the token count in the sampler context
    using llama_token_cnt = std::unordered_map<llama_token, int>;
    llama_token_cnt token_count;

    for (int i = 0; i < std::min<int>(m_penalty_last_n, m_prev.size()); ++i) {
        token_count[m_prev[i]]++;
    }

    // Apply frequency and presence penalties to the cur_p
    for (size_t i = 0; i < probs.m_probs.size(); ++i) {
        const auto token_iter = token_count.find(probs.m_probs[i].token);
        if (token_iter == token_count.end()) {
            continue;
        }

        const int count = token_iter->second;

        // The academic publication that described this technique actually just only divided, but that would cause
        // tokens with negative logits to become more likely, which is obviously wrong. This is common fix for this
        // problem, which is to multiply by the penalty instead of dividing.
        if (probs.m_probs[i].prob <= 0) {
            probs.m_probs[i].prob *= m_penalty_repeat;
        } else {
            probs.m_probs[i].prob /= m_penalty_repeat;
        }

        probs.m_probs[i].prob -= float(count) * m_penalty_freq + float(count > 0) * m_penalty_present;
    }

    probs.m_is_sorted = false;

    if (!m_penalize_nl && nl_found) {
        // restore the logit of the newline token if it was penalized
        probs.m_probs[nl_idx].prob = nl_logit;
    }
}

void RepeatPenaltySampler::accept(Token token) {
    if (m_penalty_last_n > 0) {
        m_prev.push_back(token);
    }
}

StochasticSampler::StochasticSampler(uint64_t seed) : m_random_state(seed) {}

void StochasticSampler::apply(ProbArray &probs) {
    probs[0] = probs.stochastic_sample(m_random_state);

    probs.resize(1);
    probs[0].prob         = 1.0f;
    probs.m_is_sorted     = true;
    probs.m_is_normalized = true;
}

} // namespace powerserve
