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

#include "httplib.h"
#include "server_handler.hpp"

#include <stdexcept>
#include <string>

struct SimpleServer {
private:
    ServerContext m_server_context;

    std::unique_ptr<httplib::Server> m_server;

    std::unique_ptr<std::thread> m_server_thread;

public:
    SimpleServer(
        const std::string &model_folder, const std::string &qnn_lib_folder, const std::string &host, const int port
    );

    ~SimpleServer() noexcept = default;

public:
    void execute() {
        if (m_server_thread) {
            throw std::runtime_error("there has already been one server thread");
        }
        m_server->listen_after_bind();
        m_server->wait_until_ready();
    }

    void join() {
        if (!m_server_thread) {
            throw std::runtime_error("the server hasn't been executed");
        }
        m_server_thread->join();
    }
};

void simple_server_handler(
    const std::string &model_folder, const std::string &qnn_lib_folder, const std::string &host, const int port
);
