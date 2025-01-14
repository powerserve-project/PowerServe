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

#include "sampler/prob_array.hpp"

#include "core/logger.hpp"

namespace powerserve {

void ProbArray::normalize() {
    if (m_is_normalized) {
        return;
    }

    double sum = 0.;
    for (const auto &p : m_probs) {
        sum += p.prob;
    }

    for (auto &p : m_probs) {
        p.prob /= sum;
    }

    m_is_normalized = true;
}

void ProbArray::softmax() {
    POWERSERVE_ASSERT(m_probs.size() > 0);
    if (!m_is_sorted) {
        std::sort(m_probs.begin(), m_probs.end(), std::greater());
        m_is_sorted = true;
    }

    auto max_prob = m_probs[0].prob;

    // Sum from the smallest to largest to reduce floating point rounding errors.
    double exp_prob_sum = 0;
    for (auto it = m_probs.rbegin(); it != m_probs.rend(); it++) {
        it->prob = std::exp(it->prob - max_prob);
        exp_prob_sum += it->prob;
    }

    for (auto &p : m_probs) {
        p.prob /= exp_prob_sum;
    }

    m_is_normalized = true;
}

void ProbArray::resize(size_t size) {
    m_probs.resize(size);
}

auto ProbArray::greedy_sample() -> ProbIndex & {
    return *std::max_element(m_probs.begin(), m_probs.end());
}

} // namespace powerserve
