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

#include "concurrentqueue.h"
#include "server_handler.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>

struct LocalRequest {
    std::string body;
};

struct LocalResponse;

class LocalServer {
public:
    using TaskQueue = moodycamel::ConcurrentQueue<LocalResponse *>;

    static constexpr auto DEFAULT_TASK_INTERVAL = std::chrono::milliseconds{100};

public:
    ServerContext m_context;

    TaskQueue m_task_queue;

    std::thread m_server_thread;

    std::atomic_flag m_stop;

public:
    LocalServer(
        const std::filesystem::path &model_folder,
        const std::filesystem::path &lib_folder,
        std::chrono::milliseconds task_interval = DEFAULT_TASK_INTERVAL
    );

    ~LocalServer() noexcept;

public:
    LocalResponse *create_completion_reponse(const LocalRequest &request);

    LocalResponse *create_chat_response(const LocalRequest &request);

    LocalResponse *create_model_response(const LocalRequest &request);

    std::optional<std::string> get_response(LocalResponse *response_ptr);

    bool poll_response(LocalResponse *response_ptr) const;

    void wait_response(LocalResponse *response_ptr) const;

    void destroy_response(LocalResponse *response_ptr);
};
