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

#include "simple_server.hpp"

#include "httplib.h"
#include "openai_api.hpp"

#include <memory>
#include <string>

SimpleServer::SimpleServer(
    const std::string &model_folder, const std::string &qnn_lib_folder, const std::string &host, const int port
) :
    m_server_context(model_folder, qnn_lib_folder) {
    // set up server
    m_server = std::make_unique<httplib::Server>();
    m_server->bind_to_port(host, port);

    const auto completion_handler = [this](const httplib::Request &request, httplib::Response &response) {
        handler_completion(m_server_context, request, response);
    };
    const auto chat_handler = [this](const httplib::Request &request, httplib::Response &response) {
        handler_chat(m_server_context, request, response);
    };
    const auto model_handler = [this](const httplib::Request &request, httplib::Response &response) {
        handler_model(m_server_context, request, response);
    };

    m_server->Post("/completion", completion_handler);
    m_server->Post("/completions", completion_handler);
    m_server->Post("/v1/completions", completion_handler);

    m_server->Post("/chat/completions", chat_handler);
    m_server->Post("/v1/chat/completions", chat_handler);

    m_server->Get("/v1/models", model_handler);

    POWERSERVE_LOG_INFO("server is listening http://{}:{}", host, port);
}

void simple_server_handler(
    const std::string &model_folder, const std::string &qnn_lib_folder, const std::string &host, const int port
) {
    SimpleServer server(model_folder, qnn_lib_folder, host, port);
    server.execute();
    server.join();
}
