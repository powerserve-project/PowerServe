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

/* Reference: https://platform.openai.com/docs/api-reference/introduction  */

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "server_handler.hpp"

#include <chrono>
#include <cstddef>
#include <thread>

inline const char *MIMETYPE_JSON = "application/json; charset=utf-8";

enum class ServerTask {
    Completion,
    Embedding,
    ReRank,
    Infill,
    Cancel,
    NextResponse,
    Metrics,
    SlotSave,
    SlotRestore,
    SlotErase,
    SetLora
};

enum class ErrorType {
    InvalidRequest,
    Authentication,
    Server,
    NotFound,
    Permission,
    Unavailable,  // custom error
    NotSupported, // custom error
};

enum class ChatRole {
    System,
    User,
    Assistant
};

template <class T_Response = httplib::Response>
inline void response_normal(const nlohmann::json &message_json, T_Response &response) {
    response.set_content(message_json.dump(), MIMETYPE_JSON);
    response.status = 200;
}

template <class T_Response = httplib::Response>
inline void response_error(const std::string &message, const ErrorType type, T_Response &response) {
    std::string type_str;
    int code = 500;
    switch (type) {
    case ErrorType::InvalidRequest:
        type_str = "invalid_request_error";
        code     = 400;
        break;
    case ErrorType::Authentication:
        type_str = "authentication_error";
        code     = 401;
        break;
    case ErrorType::NotFound:
        type_str = "not_found_error";
        code     = 404;
        break;
    case ErrorType::Server:
        type_str = "server_error";
        code     = 500;
        break;
    case ErrorType::Permission:
        type_str = "permission_error";
        code     = 403;
        break;
    case ErrorType::NotSupported:
        type_str = "not_supported_error";
        code     = 501;
        break;
    case ErrorType::Unavailable:
        type_str = "unavailable_error";
        code     = 503;
        break;
    }

    const nlohmann::json response_json{{"code", code}, {"message", message}, {"type", type_str}};
    response.set_content(response_json.dump(), MIMETYPE_JSON);
    response.status = code;
}

/*!
 * @brief Parse message for completion request
 * @ref https://platform.openai.com/docs/api-reference/chat/create
 */
inline ModelInput parse_completion_model_input(const nlohmann::json &request) {
    static size_t request_counter = 0;

    ModelInput input{
        .m_model         = request["model"],
        .m_prompt        = request.value("prompt", ""),
        .m_max_num_token = request.value("max_tokens", size_t{16}),
        .m_temperature   = request.value("temperature", 1.F),
        .m_top_p         = request.value("top_p", 1.F),

        .m_presence_penalty  = request.value("presence_penalty", 0.F),
        .m_frequency_penalty = request.value("frequency_penalty", 0.F),

        .m_response_n = request.value("n", size_t{1}),
        .m_best_of_n  = request.value("best_of", size_t{1}),
        .m_log_probs  = request.value("logprobs", 0),
        .stream       = request.value("stream", false),

        .m_repeat_penalty = request.value("repeat_penalty", 1.F),

        .request_id = request_counter++
    };
    return input;
}

/*!
 * @brief Parse message for chat request
 * @ref https://platform.openai.com/docs/api-reference/chat/create
 */
inline ModelInput parse_chat_model_input(const nlohmann::json &request) {
    static size_t request_counter = 0;

    ModelInput input{
        .m_model         = request["model"],
        .m_max_num_token = request.value("max_tokens", size_t{16}),
        .m_temperature   = request.value("temperature", 1.F),
        .m_top_p         = request.value("top_p", 1.F),

        .m_presence_penalty  = request.value("presence_penalty", 0.F),
        .m_frequency_penalty = request.value("frequency_penalty", 0.F),

        .stream = request.value("stream", false),

        .m_repeat_penalty = request.value("repeat_penalty", 1.F),
        .request_id       = request_counter++
    };
    const auto &message = request["messages"];
    for (const auto &entry : message) {
        input.m_history.push_back({.role = entry["role"], .content = entry["content"]});
    }
    return input;
}

inline nlohmann::json dump_completion_response(const ModelInput &input, const ModelOutput &output) {
    const auto choices = std::vector<nlohmann::json>{
        {{"text", output.m_text}, {"index", 0}, {"logprobs", nullptr}, {"finish_reason", "stop"}}
    };

    return {
        {"id", "powerserve-completion-" + std::to_string(input.request_id)},
        {"model", input.m_model},
        {"created", std::time(0)},
        {"choices", choices},
        {"usage",
         {{"prompt_tokens", output.m_input_num_token},
          {"completion_tokens", output.m_output_num_token},
          {"total_tokens", output.m_input_num_token + output.m_output_num_token}}}
    };
}

inline nlohmann::json dump_completion_chunk_response(const ModelInput &input, const ModelOutput &output) {
    auto choices = std::vector<nlohmann::json>{{{"text", output.m_text}, {"index", 0}, {"logprobs", nullptr}}};

    if (!output.m_stop_reason.has_value()) { // unfinished
        choices[0]["finish_reason"] = nullptr;
    } else { // finished
        choices[0]["finish_reason"] = output.m_stop_reason.value();
    }

    return {
        {"id", "powerserve-completion-" + std::to_string(input.request_id)},
        {"model", input.m_model},
        {"created", std::time(0)},
        {"choices", choices}
    };
}

inline nlohmann::json dump_chat_chunk_response(const ModelInput &input, const ModelOutput &output) {
    auto choices = std::vector<nlohmann::json>{
        {{"index", 0},
         {
             "delta",
             {{"role", "assistant"}, {"content", output.m_text}},
         }}
    };

    if (!output.m_stop_reason.has_value()) { // unfinished
        choices[0]["finish_reason"] = nullptr;
    } else { // finished
        choices[0]["finish_reason"] = output.m_stop_reason.value();
    }

    return {
        {"id", "powerserve-chat-" + std::to_string(input.request_id)},
        {"model", input.m_model},
        {"object", "chat.completion.chunk"},
        {"created", std::time(0)},
        {"choices", choices}
    };
}

inline nlohmann::json dump_chat_completion_response(const ModelInput &input, const ModelOutput &output) {
    // chat completion
    const auto choices = std::vector<nlohmann::json>{
        {{"index", 0}, {"message", {{"role", "assistant"}, {"content", output.m_text}}}, {"finish_reason", "stop"}}
    };

    return {
        {"id", "powerserve-chat-" + std::to_string(input.request_id)},
        {"object", input.stream ? "chat.completion.chunk" : "chat.completion"},
        {"created", std::time(0)},
        {"choices", choices},
        {"usage",
         {{"prompt_tokens", output.m_input_num_token},
          {"completion_tokens", output.m_output_num_token},
          {"total_tokens", output.m_input_num_token + output.m_output_num_token}}}
    };
}

inline nlohmann::json dump_chat_response(const ModelInput &input, const ModelOutput &output) {
    if (input.stream) { // chat chunk completion
        return dump_chat_chunk_response(input, output);
    }
    // chat completion
    return dump_chat_completion_response(input, output);
}

inline nlohmann::json dump_model_response(const std::vector<std::string> models) {
    std::vector<nlohmann::json> model_info_list;

    for (const std::string &model : models) {
        model_info_list.push_back({{"id", model}, {"object", "model"}, {"owned_by", "PowerServe"}});
    }

    return {{"data", model_info_list}, {"object", "list"}};
}

template <
    class T_Request  = httplib::Request,
    class T_Response = httplib::Response,
    class T_DataSink = httplib::DataSink>
inline void handler_completion(ServerContext &server_context, const T_Request &request, T_Response &response) {
    POWERSERVE_LOG_INFO("process completion task");
    /* Parse received message */
    ModelInput model_input;
    try {
        const auto body = nlohmann::json::parse(request.body);
        /* Set input */
        model_input = parse_completion_model_input(body);
    } catch (const std::exception &err) {
        response_error(err.what(), ErrorType::InvalidRequest, response);
        return;
    } catch (...) {
        response_error("unknown error", ErrorType::Server, response);
        return;
    }

    try {
        if (!model_input.stream) {
            /* Inference */
            const ModelOutput model_output = completion(server_context, model_input);

            /* Send result */
            response_normal(dump_completion_response(model_input, model_output), response);
        } else {
            const auto chunked_content_provider = [&server_context, model_input](size_t, T_DataSink &sink) {
                // set up a session
                const ServerSessionId session_id = server_context.setup_session(model_input);

                // fetch result from queue
                ServerSession &session = server_context.get_session(session_id);
                session.init([&] {
                    try {
                        completion(server_context, session);
                    } catch (const std::exception &err) {
                        // Enqueue error message
                        const std::string reason = err.what();
                        session.m_result_queue.enqueue(
                            {.m_text             = reason,
                             .m_input_num_token  = 0,
                             .m_output_num_token = 0,
                             .m_stop_reason      = "exception"}
                        );
                    }
                });

                while (true) {
                    const auto result = session.fetch_result();
                    if (!result.has_value()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds{100});
                        continue;
                    }

                    const auto completion_chunk    = dump_completion_chunk_response(session.m_input, result.value());
                    const std::string response_str = fmt::format("data: {}\n\n", completion_chunk.dump());
                    sink.write(response_str.data(), response_str.size());

                    if (result->m_stop_reason.has_value()) {
                        break;
                    }
                }

                // send result
                constexpr std::string_view end_data = "data: [DONE]\n\n";
                sink.write(end_data.data(), end_data.size());
                sink.done();

                // TODO: we should keep session for future KVCache reuse
                // destroy the session
                server_context.destroy_session(session_id);

                return true;
            };
            response.set_chunked_content_provider("text/event-stream", chunked_content_provider);
        }

        POWERSERVE_LOG_INFO("after completion: {}", powerserve::perf_get_mem_result());
    } catch (const std::invalid_argument &err) {
        response_error(err.what(), ErrorType::InvalidRequest, response);
    } catch (const std::exception &err) {
        response_error(err.what(), ErrorType::Server, response);
    } catch (...) {
        response_error("unknown error", ErrorType::Server, response);
    }
}

template <
    class T_Request  = httplib::Request,
    class T_Response = httplib::Response,
    class T_DataSink = httplib::DataSink>
inline void handler_chat(ServerContext &server_context, const T_Request &request, T_Response &response) {
    POWERSERVE_LOG_INFO("process chat task");
    /* Parse received message */
    ModelInput model_input;
    try {
        const auto body = nlohmann::json::parse(request.body);
        /* Set input */
        model_input = parse_chat_model_input(body);
    } catch (const std::exception &err) {
        response_error(err.what(), ErrorType::InvalidRequest, response);
        return;
    } catch (...) {
        response_error("unknown error", ErrorType::Server, response);
        return;
    }

    try {
        if (!model_input.stream) {
            /* Inference */
            const ModelOutput model_output = chat(server_context, model_input);

            /* Send result */
            response_normal(dump_chat_response(model_input, model_output), response);

        } else {
            const auto chunked_content_provider = [&server_context, model_input](size_t, T_DataSink &sink) {
                // set up a session
                const ServerSessionId session_id = server_context.setup_session(model_input);

                // fetch result from queue
                ServerSession &session = server_context.get_session(session_id);
                session.init([&] {
                    try {
                        chat(server_context, session);
                    } catch (const std::exception &err) {
                        // Enqueue error message
                        const std::string reason = err.what();
                        session.m_result_queue.enqueue(
                            {.m_text             = reason,
                             .m_input_num_token  = 0,
                             .m_output_num_token = 0,
                             .m_stop_reason      = "exception"}
                        );
                    }
                });

                while (true) {
                    const auto result = session.fetch_result();
                    if (!result.has_value()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds{100});
                        continue;
                    }

                    const auto chat_chunk          = dump_chat_chunk_response(session.m_input, result.value());
                    const std::string response_str = fmt::format("data: {}\n\n", chat_chunk.dump());
                    sink.write(response_str.data(), response_str.size());

                    if (result->m_stop_reason.has_value()) {
                        break;
                    }
                }

                // send result
                constexpr std::string_view end_data = "data: [DONE]\n\n";
                sink.write(end_data.data(), end_data.size());
                sink.done();

                // TODO: we should keep session for future KVCache reuse
                // destroy the session
                server_context.destroy_session(session_id);

                return true;
            };
            response.set_chunked_content_provider("text/event-stream", chunked_content_provider);
        }

        POWERSERVE_LOG_INFO("after chat: {}", powerserve::perf_get_mem_result());
    } catch (const std::invalid_argument &err) {
        response_error(err.what(), ErrorType::InvalidRequest, response);
    } catch (const std::exception &err) {
        response_error(err.what(), ErrorType::Server, response);
    } catch (...) {
        response_error("unknown error", ErrorType::Server, response);
    }
}

template <class T_Request = httplib::Request, class T_Response = httplib::Response>
inline void handler_model(
    ServerContext &server_context, [[maybe_unused]] const T_Request &request, T_Response &response
) {
    POWERSERVE_LOG_INFO("process model task");
    try {
        std::vector<std::string> models = list_models(server_context);
        const auto response_json        = dump_model_response(models);
        response.set_content(response_json.dump(), MIMETYPE_JSON);
    } catch (const std::exception &err) {
        response_error(err.what(), ErrorType::Server, response);
    } catch (...) {
        response_error("unknown error", ErrorType::Server, response);
    }
}
