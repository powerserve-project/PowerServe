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

#include "core/logger.hpp"
#include "core/typedefs.hpp"

#include <random>
#include <span>

namespace powerserve {

struct ProbIndex {
    float prob  = 0.0f;
    Token token = -1;

    bool operator<(const ProbIndex &other) const {
        return prob < other.prob;
    }

    bool operator>(const ProbIndex &other) const {
        return prob > other.prob;
    }
};

struct ProbArray {
    std::vector<ProbIndex> m_probs;
    bool m_is_sorted     = false; // Is sorted in descending order?
    bool m_is_normalized = false; // Does the sum of probs equal to 1?

    ProbArray(std::span<const float> logits) {
        m_probs.resize(logits.size());
        for (size_t i = 0; i < logits.size(); i++) {
            m_probs[i].token = i;
            m_probs[i].prob  = logits[i];
        }
    }

    auto operator[](size_t i) -> ProbIndex & {
        return m_probs[i];
    }

    auto begin() {
        return m_probs.begin();
    }

    auto end() {
        return m_probs.end();
    }

    void normalize();
    void softmax();

    void resize(size_t size);

    template <typename RandomEngine>
    auto stochastic_sample(RandomEngine &&gen) -> ProbIndex & {
        POWERSERVE_ASSERT(m_is_normalized);

        size_t index = std::discrete_distribution(m_probs.size(), 0, m_probs.size(), [&](double x) {
            // https://en.cppreference.com/w/cpp/numeric/random/discrete_distribution/discrete_distribution
            // x = i + 0.5
            return m_probs[(size_t)x].prob;
        })(gen);

        return m_probs[index];
    }

    auto greedy_sample() -> ProbIndex &;
};

} // namespace powerserve
