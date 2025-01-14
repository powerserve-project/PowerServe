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

/*
 * @ref: https://platform.openai.com/docs/api-reference
 */

#include "backend/platform.hpp"
#include "concurrentqueue.h"
#include "core/config.hpp"
#include "core/logger.hpp"
#include "core/timer.hpp"
#include "core/typedefs.hpp"
#include "model/model.hpp"
#include "model/model_loader.hpp"
#include "model/module/norm_attention.hpp"
#include "sampler/sampler_chain.hpp"
#include "speculative/spec_model.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>

using ModelChatHistroyEntry = powerserve::ChatEntry;

struct ModelInput {

    /* Basic config */
    /// ID of the model to use.
    std::string m_model;

    /// [Only Completion] The prompt(s) to generate completions for, encoded as a string, array of strings, array of
    /// tokens, or array of token arrays.
    std::string m_prompt;
    /// [Only Chat] The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens,
    /// or array of token arrays.
    std::vector<ModelChatHistroyEntry> m_history;

    /// The maximum number of tokens that can be generated in the completion.
    /// The token count of your prompt plus max_tokens cannot exceed the model's context length.
    size_t m_max_num_token;

    /* Sample config */

    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,
    /// while lower values like 0.2 will make it more focused and deterministic.
    float m_temperature;
    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of
    /// the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are
    /// considered.
    float m_top_p;
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so
    /// far, increasing the model's likelihood to talk about new topics.
    float m_presence_penalty;
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text
    /// so far, decreasing the model's likelihood to repeat the same line verbatim.
    float m_frequency_penalty;

    /* Generation config */

    /// [Only Completion] How many completions to generate for each prompt.
    size_t m_response_n;
    /// [Only Completion] Generates best_of completions server-side and returns the "best" (the one with the highest log
    /// probability per token). Results cannot be streamed. When used with n, best_of controls the number of candidate
    /// completions and n specifies how many to return – best_of must be greater than n.
    size_t m_best_of_n;
    /// Include the log probabilities on the logprobs most likely output tokens, as well the chosen tokens. For example,
    /// if logprobs is 5, the API will return a list of the 5 most likely tokens. The API will always return the logprob
    /// of the sampled token, so there may be up to logprobs+1 elements in the response. The maximum value for logprobs
    /// is 5.
    int m_log_probs;
    /// Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent events as they
    /// become available, with the stream terminated by a data: [DONE] message.
    bool stream;

    /* Extension */

    float m_repeat_penalty;

    /* Metadata */

    size_t request_id;

    // unsupported: logit_bias
};

struct ModelOutput {
    std::string m_text;
    size_t m_input_num_token;
    size_t m_output_num_token;
    std::optional<std::string> m_stop_reason;
};

using ServerSessionId = int;

struct ServerSession {
    using ResultQueue = moodycamel::ConcurrentQueue<ModelOutput>;

    static constexpr int MAX_CHUNK_SIZE = 16;

    static constexpr int MAX_NUM_GENERATOR = 1;

    static constexpr int MAX_NUM_CONSUMER = 1;

public:
    ModelInput m_input;

    ResultQueue m_result_queue;

    std::unique_ptr<std::thread> m_session_thread_ptr;

public:
    ServerSession() = default;

    ServerSession(const ModelInput &input) :
        m_input(input),
        m_result_queue(MAX_CHUNK_SIZE, MAX_NUM_GENERATOR, MAX_NUM_CONSUMER) {}

    ~ServerSession() noexcept {
        if (m_session_thread_ptr) {
            m_session_thread_ptr->join();
        }
    }

    ServerSession(ServerSession &&other) noexcept = default;

    ServerSession &operator=(ServerSession &&other) noexcept = default;

public:
    void init(std::function<void()> &&thread_func) {
        if (m_session_thread_ptr) {
            POWERSERVE_LOG_ERROR("trying to init a session twice");
        } else {
            m_session_thread_ptr = std::make_unique<std::thread>(std::move(thread_func));
        }
    }

    std::optional<ModelOutput> fetch_result() {
        std::vector<ModelOutput> output_array(MAX_CHUNK_SIZE);
        const size_t actual_num = m_result_queue.try_dequeue_bulk(output_array.begin(), MAX_CHUNK_SIZE);
        output_array.resize(actual_num);

        // merge outputs
        if (actual_num == 0) {
            return std::nullopt;
        }

        ModelOutput output{
            .m_text             = {},
            .m_input_num_token  = output_array.back().m_input_num_token,
            .m_output_num_token = output_array.back().m_output_num_token,
            .m_stop_reason      = output_array.back().m_stop_reason
        };
        for (const ModelOutput &entry : output_array) {
            output.m_text += entry.m_text;
        }

        return output;
    }
};

struct ModelContext {
public:
    powerserve::Config m_config;
    std::shared_ptr<powerserve::Model> m_model_ptr;
    std::shared_ptr<powerserve::Model> m_draft_model_ptr;
    std::unique_ptr<powerserve::Tokenizer> m_tokenizer_ptr;
    powerserve::SpeculativeConfig speculative_config;

public:
    ModelContext() = default;

    ModelContext(
        const powerserve::Config &config,
        std::shared_ptr<powerserve::Model> &&model_ptr,
        std::unique_ptr<powerserve::Tokenizer> &&tokenizer_ptr
    ) :
        m_config(config),
        m_model_ptr(std::move(model_ptr)),
        m_tokenizer_ptr(std::move(tokenizer_ptr)) {}

    ModelContext(
        const powerserve::Config &config,
        std::shared_ptr<powerserve::Model> &&model_ptr,
        std::shared_ptr<powerserve::Model> &&draft_model_ptr,
        std::unique_ptr<powerserve::Tokenizer> &&tokenizer_ptr
    ) :
        m_config(config),
        m_model_ptr(std::move(model_ptr)),
        m_draft_model_ptr(std::move(draft_model_ptr)),
        m_tokenizer_ptr(std::move(tokenizer_ptr)) {}

    ~ModelContext() noexcept = default;

    ModelContext(const ModelContext &other) = delete;

    ModelContext(ModelContext &&other) noexcept = default;

    ModelContext &operator=(const ModelContext &other) = delete;

    ModelContext &operator=(ModelContext &&other) noexcept = default;
};

struct ServerContext {
private:
    std::filesystem::path m_work_folder;

    std::filesystem::path m_lib_folder;

    std::mutex m_lock;

    std::shared_ptr<powerserve::Platform> m_platform_ptr;

    std::unordered_map<std::string, std::shared_ptr<powerserve::Model>> m_model_map;

    std::unordered_map<std::string, ModelContext> m_context_slot_map;

    std::map<ServerSessionId, ServerSession> m_session_map;

public:
    ServerContext(powerserve::Path work_folder, powerserve::Path lib_folder) :
        m_work_folder(std::move(work_folder)),
        m_lib_folder(std::move(lib_folder)),
        m_platform_ptr(std::make_shared<powerserve::Platform>()) {
        if (!std::filesystem::exists(m_work_folder)) {
            POWERSERVE_LOG_WARN("model base folder does not exist: {}", m_work_folder);
        }
        if (!std::filesystem::is_directory(m_work_folder)) {
            POWERSERVE_LOG_WARN("model base folder is not directory: {}", m_work_folder);
        }

#if defined(POWERSERVE_WITH_QNN)
        m_platform_ptr->init_qnn_backend(m_lib_folder);
#endif // POWERSERVE_WITH_QNN
    }

    ~ServerContext() = default;

public:
    ModelContext &setup_model(const std::string &model_name) {
        // Parse model name
        std::string_view main_model_name;
        std::string_view draft_model_name;
        {
            const auto iter = model_name.find('+');
            if (iter == std::string::npos) {
                main_model_name = std::string_view(model_name);
            } else {
                main_model_name  = std::string_view(model_name.cbegin(), model_name.cbegin() + iter);
                draft_model_name = std::string_view(model_name.cbegin() + iter + 1, model_name.cend());
            }
        }
        POWERSERVE_LOG_INFO("main model: {}, draft model: {}", main_model_name, draft_model_name);

        std::lock_guard<std::mutex> lock_guard(m_lock);

        if (m_context_slot_map.contains(model_name)) {
            POWERSERVE_LOG_INFO("found cached model context: {}", model_name);
            return m_context_slot_map.at(model_name);
        }

#ifndef POWERSERVE_SERVER_MULTIMODEL
        destroy_all_models_unsafe();
#endif // POWERSERVE_SERVER_MULTIMODEL

        const powerserve::Path main_model_path  = model_name_to_path(main_model_name);
        const powerserve::Path draft_model_path = draft_model_name.empty() ? "" : model_name_to_path(draft_model_name);

        const powerserve::Config workspace_config(
            m_work_folder, powerserve::Path(m_work_folder) / powerserve::WORKSPACE_CONFIG_FILENAME
        );

        const std::string tokenizer_path                     = main_model_path / powerserve::MODEL_VOCAB_FILENAME;
        std::unique_ptr<powerserve::Tokenizer> tokenizer_ptr = std::make_unique<powerserve::Tokenizer>(tokenizer_path);
        POWERSERVE_LOG_INFO("after tokenizer init: {}", powerserve::perf_get_mem_result());

        std::shared_ptr<powerserve::Model> main_model = init_model(main_model_path, workspace_config.hyper_params);
        if (draft_model_path.empty()) {
            m_context_slot_map[model_name] =
                ModelContext(workspace_config, std::move(main_model), std::move(tokenizer_ptr));
        } else {
            std::shared_ptr<powerserve::Model> draft_model =
                init_model(draft_model_path, workspace_config.hyper_params);
            m_context_slot_map[model_name] =
                ModelContext(workspace_config, std::move(main_model), std::move(draft_model), std::move(tokenizer_ptr));
        }

        return m_context_slot_map.at(model_name);
    }

    void destroy_model(const std::string &model_name) {
        std::lock_guard<std::mutex> lock_guard(m_lock);
        if (!m_model_map.contains(model_name)) {
            return;
        }

        // Unload model from backends
        const auto &model = m_model_map[model_name];
        m_platform_ptr->destroy_ggml_backend(model->m_config);

#if defined(POWERSERVE_WITH_QNN)
        m_platform_ptr->qnn_backend->unload_model(model->m_config);
#endif // POWERSERVE_WITH_QNN

        // Erase model and model context
        m_context_slot_map.erase(model_name);
        m_model_map.erase(model_name);
    }

    void destroy_all_models() {
        std::lock_guard<std::mutex> lock_guard(m_lock);
        destroy_all_models_unsafe();
    }

    std::vector<std::string> list_models() const {
        if (!std::filesystem::exists(m_work_folder) && !std::filesystem::is_directory(m_work_folder)) {
            POWERSERVE_LOG_ERROR("model base folder does not exist: {}", m_work_folder);
            return {};
        }

        std::vector<std::string> model_list;
        for (const auto &entry : std::filesystem::directory_iterator(m_work_folder)) {
            const std::string dir_name = entry.path().filename();

            // TODO: no hardcoded string
            if (dir_name == "bin" || dir_name == "qnn_libs") {
                continue;
            }

            if (!entry.is_directory()) {
                POWERSERVE_LOG_ERROR("model folder is not directory: {}", m_work_folder);
                continue;
            }
            model_list.emplace_back(entry.path().filename());
        }

        return model_list;
    }

public:
    ServerSessionId setup_session(const ModelInput &input) {
        static int counter = 0;
        std::lock_guard<std::mutex> lock_guard(m_lock);
        while (true) {
            const int new_id = counter++;
            if (m_session_map.contains(new_id)) {
                continue;
            }
            m_session_map[new_id] = ServerSession(input);

            POWERSERVE_LOG_INFO("set up session: {}", new_id);
            return new_id;
        }
    }

    ServerSession &get_session(const ServerSessionId session_id) {
        std::lock_guard<std::mutex> lock_guard(m_lock);
        return m_session_map[session_id];
    }

    void destroy_session(const ServerSessionId session_id) {
        {
            std::lock_guard<std::mutex> lock_guard(m_lock);
            if (!m_session_map.contains(session_id)) {
                POWERSERVE_LOG_WARN("cannot destroy session with session id: {}", session_id);
                return;
            }
            m_session_map.erase(session_id);
        }
        POWERSERVE_LOG_INFO("destroy session: {}", session_id);
    }

private:
    powerserve::Path model_name_to_path(const std::string_view model_name) const {
        const powerserve::Path inner_model_folder = m_work_folder / model_name;
        powerserve::Path model_folder;
        if (std::filesystem::exists(inner_model_folder) && std::filesystem::is_directory(inner_model_folder)) {
            model_folder = inner_model_folder;
        } else if (std::filesystem::exists(model_name) && std::filesystem::is_directory(model_name)) {
            model_folder = model_name;
        } else {
            POWERSERVE_LOG_ERROR("model folder does not exist: {}", model_name);
            throw std::invalid_argument("model folder does not exist");
        }
        POWERSERVE_LOG_INFO("found model folder: {}", model_folder);
        return model_folder;
    }

    std::shared_ptr<powerserve::Model> init_model(
        const powerserve::Path &model_path, const powerserve::HyperParams &hyper_params
    ) {
        if (m_model_map.contains(model_path)) {
            return m_model_map.at(model_path);
        }

        std::shared_ptr<powerserve::Model> model_ptr = powerserve::load_model(model_path);

        model_ptr->m_platform = m_platform_ptr;
        m_platform_ptr->init_ggml_backend(model_ptr->m_config, hyper_params);

        model_ptr->m_attn = std::make_shared<powerserve::NormAttention>(model_ptr->m_config->llm, model_ptr->m_weights);
        POWERSERVE_LOG_INFO("after attn init: {}", powerserve::perf_get_mem_result());

#if defined(POWERSERVE_WITH_QNN)
        m_platform_ptr->qnn_backend->load_model(
            model_path / powerserve::qnn::QNN_WORKSPACE_DIR_NAME, model_ptr->m_config
        );
        model_ptr->kv_cache =
            model_ptr->m_platform->qnn_backend->m_models[model_ptr->m_config->model_id]->kv_cache.get();
#endif // POWERSERVE_WITH_QNN

        m_model_map[model_path] = model_ptr;
        return model_ptr;
    }

    void destroy_all_models_unsafe() {
        POWERSERVE_LOG_INFO("Destory all models in memory");
        for (const auto &[_, model] : m_model_map) {
            m_platform_ptr->destroy_ggml_backend(model->m_config);
#if defined(POWERSERVE_WITH_QNN)
            m_platform_ptr->qnn_backend->unload_model(model->m_config);
#endif // POWERSERVE_WITH_QNN
        }

        m_context_slot_map.clear();
        m_model_map.clear();
    }
};

/*!
 * @param output_string[in] The output string after tokenization of model
 * @note For some reasons(e.g. truncation), the output token may be incomplete. In case of json parser exception,
 * we need to hold the incomplete word until next time or the end.
 */
inline bool is_utf8_string_incomplete(const std::string &output_string) {
    bool incomplete = false;
    for (unsigned i = 1; i < 5 && i <= output_string.size(); ++i) {
        unsigned char c = output_string[output_string.size() - i];
        if ((c & 0xC0) == 0x80) {
            // continuation byte: 10xxxxxx
            continue;
        }
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character: 110xxxxx ...
            incomplete = i < 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character: 1110xxxx ...
            incomplete = i < 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character: 11110xxx ...
            incomplete = i < 4;
        }
        // else 1-byte character or invalid byte
        break;
    }
    return incomplete;
}

inline std::string &remove_incomplete_utf8_char(std::string &output_string) {
    for (unsigned i = 1; i < 5 && i <= output_string.size(); ++i) {
        unsigned char c = output_string[output_string.size() - i];
        if ((c & 0xC0) == 0x80) {
            // continuation byte: 10xxxxxx
            continue;
        }
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character: 110xxxxx ...
            if (i < 2) {
                output_string.erase(output_string.size() - i, i);
                return output_string;
            };
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character: 1110xxxx ...
            if (i < 3) {
                output_string.erase(output_string.size() - i, i);
                return output_string;
            };
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character: 11110xxx ...
            if (i < 4) {
                output_string.erase(output_string.size() - i, i);
                return output_string;
            };
        }
        // else 1-byte character or invalid byte
        break;
    }
    POWERSERVE_LOG_INFO("The output string is completed");
    return output_string;
}

inline void stream_inference(const ModelContext &context, ServerSession &session, const std::string &input_prompt) {
    using namespace powerserve;

    const ModelInput &input = session.m_input;

    auto &config    = context.m_config;
    auto &model     = *context.m_model_ptr;
    auto &draft     = context.m_draft_model_ptr;
    auto &tokenizer = *context.m_tokenizer_ptr;

    // TODO: This sampler config is too argly
    HyperParams::SamplerConfig sampler_config = config.hyper_params.sampler_config;
    sampler_config.temperature                = input.m_temperature;
    sampler_config.penalty_freq               = input.m_frequency_penalty;
    sampler_config.penalty_present            = input.m_presence_penalty;
    sampler_config.penalty_repeat             = input.m_repeat_penalty;
    sampler_config.top_p                      = input.m_top_p;
    sampler_config.temperature                = input.m_temperature;
    powerserve::SamplerChain sampler{sampler_config, tokenizer};

    /* Inference */
    ModelOutput output;

    const size_t max_num_token = input.m_max_num_token;
    const size_t batch_size    = config.hyper_params.batch_size;

    std::string stop_reason = "length";
    size_t step             = 0;

    POWERSERVE_LOG_DEBUG("Model input     : {}", powerserve::abbreviation(input_prompt, 50));
    POWERSERVE_LOG_DEBUG("Model max token : {}", max_num_token);
    POWERSERVE_LOG_DEBUG("Model batch size: {}", batch_size);

    /*
     * Prefill
     */
    Timer timer;
    const size_t num_prefill_token = tokenizer.tokenize(input_prompt, tokenizer.m_vocab.tokenizer_add_bos).size() - 1;

    bool end_of_text = false;
    std::string output_buffer;

    std::shared_ptr<powerserve::TokenIterator> iter = nullptr;
#if defined(POWERSERVE_WITH_QNN)
    std::shared_ptr<powerserve::SpeculativeModel> spec_model = nullptr;
    if (draft) {
        spec_model = std::make_shared<powerserve::SpeculativeModel>(
            context.m_model_ptr, context.m_draft_model_ptr, context.speculative_config
        );
        iter = spec_model->generate(tokenizer, sampler, input_prompt, max_num_token, batch_size);
    } else
#endif
    {
        iter = model.generate(tokenizer, sampler, input_prompt, max_num_token, batch_size);
    }
    const size_t prefill_time_ms = timer.elapsed_time_ms();

    while (!iter->end()) {
        auto token = iter->next();
        step++;
        if (step == 1) {
            POWERSERVE_LOG_INFO(
                "prefill step: {}, prefill time: {}ms ({} token/s)",
                num_prefill_token,
                prefill_time_ms,
                num_prefill_token * 1000.f / prefill_time_ms
            );
            timer.reset();
            continue;
        } // Avoid outputting the last token

        if (token == tokenizer.bos_token()) {
            continue;
        }

        if (tokenizer.should_stop(token)) {
            end_of_text = true;
            break;
        } else {
            output_buffer += tokenizer.to_string(token);
            if (!is_utf8_string_incomplete(output_buffer)) {
                session.m_result_queue.enqueue(
                    {.m_text             = output_buffer,
                     .m_input_num_token  = num_prefill_token,
                     .m_output_num_token = step,
                     .m_stop_reason      = std::nullopt}
                );
                output_buffer.clear();
            }
        }
    }

    const std::string_view end_text = end_of_text ? "[end of text]" : "";

    session.m_result_queue.enqueue(
        {.m_text             = remove_incomplete_utf8_char(output_buffer).append(end_text),
         .m_input_num_token  = num_prefill_token,
         .m_output_num_token = step,
         .m_stop_reason      = end_of_text ? "stop" : "length"}
    );

    const size_t decode_time_ms = timer.elapsed_time_ms();
    POWERSERVE_LOG_INFO(
        "decode  step: {}, decode  time: {}ms ({} token/s)", step, decode_time_ms, step * 1000.f / decode_time_ms
    );
}

inline ModelOutput blocking_inference(
    const ModelContext &context, const ModelInput &input, const std::string &input_prompt
) {
    using namespace powerserve;

    auto &config    = context.m_config;
    auto &model     = *context.m_model_ptr;
    auto &draft     = context.m_draft_model_ptr;
    auto &tokenizer = *context.m_tokenizer_ptr;

    auto sampler_config            = config.hyper_params.sampler_config;
    sampler_config.temperature     = input.m_temperature;
    sampler_config.penalty_freq    = input.m_frequency_penalty;
    sampler_config.penalty_present = input.m_presence_penalty;
    sampler_config.penalty_repeat  = input.m_repeat_penalty;
    sampler_config.top_p           = input.m_top_p;
    sampler_config.temperature     = input.m_temperature;
    powerserve::SamplerChain sampler{sampler_config, tokenizer};

    /* Inference */
    ModelOutput output;

    const size_t max_num_token = input.m_max_num_token;
    const size_t batch_size    = config.hyper_params.batch_size;

    std::string output_text;
    std::string stop_reason = "length";
    size_t step             = 0;

    POWERSERVE_LOG_DEBUG("Model input     : {}", powerserve::abbreviation(input_prompt, 20));
    POWERSERVE_LOG_DEBUG("Model max token : {}", max_num_token);
    POWERSERVE_LOG_DEBUG("Model batch size: {}", batch_size);

    /*
     * Prefill
     */
    Timer timer;
    const size_t num_prefill_token = tokenizer.tokenize(input_prompt, tokenizer.m_vocab.tokenizer_add_bos).size() - 1;
    bool end_of_text               = false;

    std::shared_ptr<powerserve::TokenIterator> iter = nullptr;
#if defined(POWERSERVE_WITH_QNN)
    std::shared_ptr<powerserve::SpeculativeModel> spec_model = nullptr;
    if (draft) {
        spec_model = std::make_shared<powerserve::SpeculativeModel>(
            context.m_model_ptr, context.m_draft_model_ptr, context.speculative_config
        );
        iter = spec_model->generate(tokenizer, sampler, input_prompt, max_num_token, batch_size);
    } else
#endif
    {
        iter = model.generate(tokenizer, sampler, input_prompt, max_num_token, batch_size);
    }
    const size_t prefill_time_ms = timer.elapsed_time_ms();

    while (!iter->end()) {
        auto token = iter->next();
        step++;
        if (step == 1) {
            POWERSERVE_LOG_INFO(
                "prefill step: {}, prefill time: {}ms ({} token/s)",
                num_prefill_token,
                prefill_time_ms,
                num_prefill_token * 1000.f / prefill_time_ms
            );
            timer.reset();
            continue;
        } // Avoid outputting the last token

        if (token == tokenizer.bos_token()) {
            continue;
        }

        if (tokenizer.should_stop(token)) {
            end_of_text = true;
            stop_reason = "stop";
            break;
        } else {
            output_text += tokenizer.to_string(token);
        }
    }

    remove_incomplete_utf8_char(output_text);
    output_text += end_of_text ? "[end of text]" : "";

    const size_t decode_time_ms = timer.elapsed_time_ms();
    POWERSERVE_LOG_INFO(
        "decode  step: {}, decode  time: {}ms ({} token/s)", step, decode_time_ms, step * 1000.f / decode_time_ms
    );
    POWERSERVE_LOG_DEBUG("Model output token: {}", output_text);

    return {
        .m_text             = output_text,
        .m_input_num_token  = num_prefill_token,
        .m_output_num_token = step,
        .m_stop_reason      = stop_reason
    };
}

/*!
 * @brief Generate
 * @param[inout] context
 * @todo Streamly generation
 */
inline ModelOutput completion(ServerContext &server_context, const ModelInput &input) {
    using namespace powerserve;
    /* Parse and concat user inputs */
    const ModelContext &context = server_context.setup_model(input.m_model);
    const Tokenizer &tokenizer  = *context.m_tokenizer_ptr;

    return blocking_inference(context, input, input.m_prompt);
}

inline void completion(ServerContext &server_context, ServerSession &session) {
    using namespace powerserve;
    /* Parse and concat user inputs */
    const ModelInput &input     = session.m_input;
    const ModelContext &context = server_context.setup_model(input.m_model);
    const Tokenizer &tokenizer  = *context.m_tokenizer_ptr;

    stream_inference(context, session, input.m_prompt);
}

inline ModelOutput chat(ServerContext &server_context, const ModelInput &input) {
    using namespace powerserve;
    /* Parse and concat user inputs */
    const ModelContext &context    = server_context.setup_model(input.m_model);
    const Tokenizer &tokenizer     = *context.m_tokenizer_ptr;
    const std::string input_prompt = tokenizer.apply_chat_template(input.m_history, true);

    return blocking_inference(context, input, input_prompt);
}

inline void chat(ServerContext &server_context, ServerSession &session) {
    using namespace powerserve;
    /* Parse and concat user inputs */
    const ModelInput &input        = session.m_input;
    const ModelContext &context    = server_context.setup_model(input.m_model);
    const Tokenizer &tokenizer     = *context.m_tokenizer_ptr;
    const std::string input_prompt = tokenizer.apply_chat_template(input.m_history, true);

    stream_inference(context, session, input_prompt);
}

inline std::vector<std::string> list_models(ServerContext &server_context) {
    return server_context.list_models();
}
