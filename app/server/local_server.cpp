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

#include "local_server.hpp"

#include "openai_api.hpp"
#include "server_handler.hpp"

struct LocalDataSink {
    using DataQueue = moodycamel::ConcurrentQueue<std::string, moodycamel::ConcurrentQueueDefaultTraits>;

    static constexpr int MAX_QUEUE_DEPTH = 16;

    static constexpr int MAX_NUM_PROVIDER = 1;

    static constexpr int MAX_NUM_COMSUMER = 1;

private:
    std::atomic_flag m_done{true};
    DataQueue m_queue;

public:
    LocalDataSink() : m_queue(MAX_QUEUE_DEPTH, MAX_NUM_PROVIDER, MAX_NUM_COMSUMER) {}

public: /* Server interface */
    void write(const char *data, const size_t size) {
        const std::string new_content(data, size);
        m_queue.enqueue(new_content);
    }

    void reset() {
        m_done.clear();
    }

    void done() {
        m_done.test_and_set();
        m_done.notify_all();
    }

public: /* Client interface */
    std::optional<std::string> fetch() {
        std::string ret;
        if (m_queue.try_dequeue(ret)) {
            return ret;
        }
        return std::nullopt;
    }

    bool poll() const {
        return m_done.test();
    }

    void wait() const {
        m_done.wait(false);
    }
};

struct LocalResponse {
public:
    int status = 200;
    LocalDataSink m_data_sink;

    std::function<void(LocalResponse &)> m_func;

public:
    template <class T>
    LocalResponse(T &&func) : m_func(std::forward<T>(func)) {}

    ~LocalResponse() noexcept = default;

    LocalResponse(const LocalResponse &other) = delete;

    LocalResponse(LocalResponse &&other) noexcept = delete;

public:
    void set_content(const std::string &content, [[maybe_unused]] const char *type) {
        m_data_sink.write(content.data(), content.size());
        m_data_sink.done();
    }

    template <class T_Provide>
    void set_chunked_content_provider(const std::string &content_type, T_Provide &&provider) {
        T_Provide data_chunk_provider(std::forward<T_Provide>(provider));
        data_chunk_provider(0, m_data_sink);
    }

public:
    void reset() {
        status = 200;
        m_data_sink.reset();
    }
};

LocalServer::LocalServer(
    const std::filesystem::path &model_folder,
    const std::filesystem::path &qnn_lib_folder,
    std::chrono::milliseconds task_interval
) :
    m_context(model_folder, qnn_lib_folder),
    m_server_thread([this, task_interval] {
        POWERSERVE_LOG_INFO("Local Server pool thread start");
        while (!m_stop.test()) {
            LocalResponse *response_ptr = nullptr;

            if (m_task_queue.try_dequeue(response_ptr)) {
                POWERSERVE_LOG_INFO("Local Server process a new task");
                response_ptr->m_func(*response_ptr);
            } else {
                std::this_thread::sleep_for(task_interval);
            }
        }
        POWERSERVE_LOG_INFO("Local Server pool thread stop");
    }) {}

LocalServer::~LocalServer() noexcept {
    m_stop.test_and_set();
    m_stop.notify_all();
    m_server_thread.join();
}

LocalResponse *LocalServer::create_completion_reponse(const LocalRequest &request) {
    LocalResponse *new_response = new LocalResponse([this, request](LocalResponse &response) {
        handler_completion<LocalRequest, LocalResponse, LocalDataSink>(m_context, request, response);
    });
    new_response->reset();
    m_task_queue.enqueue(new_response);

    return new_response;
}

LocalResponse *LocalServer::create_chat_response(const LocalRequest &request) {
    LocalResponse *new_response = new LocalResponse([this, request](LocalResponse &response) {
        handler_chat<LocalRequest, LocalResponse, LocalDataSink>(m_context, request, response);
    });
    new_response->reset();
    m_task_queue.enqueue(new_response);

    return new_response;
}

LocalResponse *LocalServer::create_model_response(const LocalRequest &request) {
    LocalResponse *new_response = new LocalResponse([this, request](LocalResponse &response) {
        handler_model<LocalRequest, LocalResponse>(m_context, request, response);
    });
    new_response->reset();
    m_task_queue.enqueue(new_response);

    return new_response;
}

std::optional<std::string> LocalServer::get_response(LocalResponse *response_ptr) {
    return response_ptr->m_data_sink.fetch();
}

bool LocalServer::poll_response(LocalResponse *response_ptr) const {
    return response_ptr->m_data_sink.poll();
}

void LocalServer::wait_response(LocalResponse *response_ptr) const {
    response_ptr->m_data_sink.wait();
}

void LocalServer::destroy_response(LocalResponse *response_ptr) {
    delete response_ptr;
}
