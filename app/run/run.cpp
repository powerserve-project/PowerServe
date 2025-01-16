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

#include "cmdline.hpp"
#include "core/logger.hpp"
#include "core/timer.hpp"
#include "model/model_loader.hpp"
#include "model/module/norm_attention.hpp"
#include "sampler/sampler_chain.hpp"
#include "speculative/spec_model.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>

int main(int argc, char *argv[]) {
    const powerserve::CommandLineArgument args = powerserve::parse_command_line("PowerServe CLI", argc, argv);
    const powerserve::Config config            = powerserve::get_config_from_argument(args);

    std::shared_ptr<powerserve::Model> main_model  = powerserve::load_model(config.main_model_dir);
    std::shared_ptr<powerserve::Model> draft_model = nullptr;
    if (args.use_spec) {
        draft_model = powerserve::load_model(config.draft_model_dir);
    }
    POWERSERVE_LOG_INFO("after model init: {}", powerserve::perf_get_mem_result());

    const auto [sampler_config, n_threads, batch_size] = config.hyper_params;
    main_model->m_platform                             = std::make_shared<powerserve::Platform>();
    auto &platform                                     = main_model->m_platform;

    platform->init_ggml_backend(main_model->m_config, config.hyper_params);

    if (args.use_spec) {
        draft_model->m_platform = platform;
        platform->init_ggml_backend(draft_model->m_config, config.hyper_params);
    }

#if defined(POWERSERVE_WITH_QNN)
    if (!args.no_qnn) {
        auto &qnn_backend = main_model->m_platform->qnn_backend;
        main_model->m_platform->init_qnn_backend(args.qnn_lib_folder);
        qnn_backend->load_model(config.main_model_dir / powerserve::qnn::QNN_WORKSPACE_DIR_NAME, main_model->m_config);
        main_model->kv_cache = platform->qnn_backend->m_models[main_model->m_config->model_id]->kv_cache.get();

        if (args.use_spec) {
            qnn_backend->load_model(
                config.draft_model_dir / powerserve::qnn::QNN_WORKSPACE_DIR_NAME, draft_model->m_config
            );
            draft_model->kv_cache = platform->qnn_backend->m_models[draft_model->m_config->model_id]->kv_cache.get();
        }
    }
#endif
    POWERSERVE_LOG_INFO("after platform init: {}", powerserve::perf_get_mem_result());

    main_model->m_attn = std::make_shared<powerserve::NormAttention>(main_model->m_config->llm, main_model->m_weights);
    if (args.use_spec) {
        draft_model->m_attn =
            std::make_shared<powerserve::NormAttention>(draft_model->m_config->llm, draft_model->m_weights);
    }
    POWERSERVE_LOG_INFO("after attn init: {}", powerserve::perf_get_mem_result());

    const std::string tokenizer_path = config.main_model_dir / powerserve::MODEL_VOCAB_FILENAME;
    powerserve::Tokenizer tokenizer(tokenizer_path);
    POWERSERVE_LOG_INFO("after tokenizer init: {}", powerserve::perf_get_mem_result());

    powerserve::SamplerChain sampler{sampler_config, tokenizer};
    POWERSERVE_LOG_INFO("after sampler init: {}", powerserve::perf_get_mem_result());

    std::string prompt = args.prompt;
    if (args.use_chat_template) {
        std::vector<powerserve::ChatEntry> chat = {{"user", args.prompt}};
        prompt                                  = tokenizer.apply_chat_template(chat, true);
    }

    {
        POWERSERVE_LOG_INFO("prompt      : {:?}", powerserve::abbreviation(prompt, 50));
        POWERSERVE_LOG_INFO("n_predicts  : {}", args.num_predict);
        POWERSERVE_LOG_INFO("model arch  : {}", main_model->m_config->arch);
        POWERSERVE_LOG_INFO("n_threads   : {}", n_threads);
        POWERSERVE_LOG_INFO("batch_size   : {}", batch_size);
    }

    // generate
    long prefill_start = 0;
    long prefill_end   = 0;
    long decode_end    = 0;
    bool start         = false;
    int actual_predict = 0;
    for (const powerserve::Token prompt_token : tokenizer.tokenize(prompt, tokenizer.m_vocab.tokenizer_add_bos)) {
        fmt::print("{}", tokenizer.to_string(prompt_token, false));
    }
    prefill_start = powerserve::timestamp_ms();

    std::shared_ptr<powerserve::TokenIterator> iter = nullptr;
#if defined(POWERSERVE_WITH_QNN)
    std::shared_ptr<powerserve::SpeculativeModel> spec_model = nullptr;
    if (args.use_spec) {
        spec_model = std::make_shared<powerserve::SpeculativeModel>(main_model, draft_model, args.speculative_config);
        iter       = spec_model->generate(tokenizer, sampler, prompt, args.num_predict, batch_size);
    } else
#endif
    {
        iter = main_model->generate(tokenizer, sampler, prompt, args.num_predict, batch_size);
    }
    prefill_end = powerserve::timestamp_ms();

    while (!iter->end()) {
        auto next = iter->next();
        if (!start) {
            start = true;
            continue;
        }
        actual_predict += 1;
        if (next == tokenizer.bos_token()) {
            break;
        }
        if (tokenizer.should_stop(next)) {
            fmt::print("[end of text]");
            break;
        }
        fmt::print("{}", tokenizer.to_string(next, false));
        fflush(stdout);
    }
    fmt::println("");

    if (start) {
        decode_end               = powerserve::timestamp_ms();
        const size_t num_prefill = tokenizer.tokenize(prompt, tokenizer.m_vocab.tokenizer_add_bos).size() - 1;
        POWERSERVE_LOG_INFO("prefill time: {} s", (double)(prefill_end - prefill_start) / 1000);
        POWERSERVE_LOG_INFO(
            "prefill speed ({} tokens): {} tokens/s",
            num_prefill,
            num_prefill / (double)(prefill_end - prefill_start) * 1000
        );
        POWERSERVE_LOG_INFO(
            "decode speed ({} tokens): {} tokens/s",
            actual_predict,
            actual_predict / (double)(decode_end - prefill_end) * 1000
        );
        POWERSERVE_LOG_INFO(
            "total speed: {} tokens/s", (num_prefill + actual_predict) / (double)(decode_end - prefill_start) * 1000
        );
    }
#if defined(POWERSERVE_WITH_QNN)
    if (args.use_spec) {
        spec_model->print_stat();
    }
#endif

    return 0;
}
