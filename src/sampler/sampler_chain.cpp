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

#include "sampler_chain.hpp"

namespace powerserve {

void SamplerChain::build_from_config(const HyperParams::SamplerConfig &config, const Tokenizer &tokenizer) {
    auto seed = config.seed;
    if (seed == (uint64_t)-1) {
        std::random_device rd;
        seed = rd();
    }
    POWERSERVE_LOG_INFO("seed: {}", seed);

    // Samplers in order:
    // - Repeat penalty
    // - Top K
    // - Temperature
    // - Top P
    // - Stochastic

    append<RepeatPenaltySampler>(
        tokenizer.n_vocabs(),
        tokenizer.m_vocab.special_eos_id,
        tokenizer.m_vocab.linefeed_id,
        config.penalty_last_n,
        config.penalty_repeat,
        config.penalty_freq,
        config.penalty_present,
        config.penalize_nl,
        config.ignore_eos
    );
    append<TopKSampler>(config.top_k);
    append<TemperatureSampler>(config.temperature);
    append<SoftmaxSampler>();
    append<TopPSampler>(config.top_p);
    append<NormalizeSampler>();
    append<StochasticSampler>(seed);
}

void SamplerChain::apply(ProbArray &probs) {
    for (auto &sampler : m_samplers) {
        sampler->apply(probs);
    }
}

void SamplerChain::accept(Token token) {
    for (auto &sampler : m_samplers) {
        sampler->accept(token);
    }
}

} // namespace powerserve
