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

#include "thread_pool.hpp"

#include "core/logger.hpp"

#include <cstring>

namespace powerserve {

ThreadPool::ThreadPool(const std::vector<ThreadConfig> &configs) : m_configs(configs) {
    POWERSERVE_ASSERT(configs.size() > 0);

    // uv_barrier_init(&m_run_barrier, 1 + size()); // 1 for main thread, size() for other threads
    // uv_barrier_init(&m_sync_barrier, size());
    spin_barrier_init(&m_run_barrier, 1 + size());
    spin_barrier_init(&m_sync_barrier, size());

    m_threads.reserve(size());
    for (size_t i = 0; i < size(); i++) {
        m_threads.emplace_back(&ThreadPool::thread_main, this, i);
    }
}

ThreadPool::~ThreadPool() {
    m_exited = true;
    // uv_barrier_wait(&m_run_barrier);
    spin_barrier_wait(&m_run_barrier);

    for (auto &thread : m_threads) {
        thread.join();
    }
    // uv_barrier_destroy(&m_run_barrier);
    // uv_barrier_destroy(&m_sync_barrier);
}

void ThreadPool::barrier() {
    // uv_barrier_wait(&m_sync_barrier);
    spin_barrier_wait(&m_sync_barrier);
}

void ThreadPool::run(TaskFn task) { // main thread entry point
    async_run(task);
    wait();
}

void ThreadPool::async_run(TaskFn task) {
    POWERSERVE_ASSERT(m_current_task == nullptr);
    m_current_task = task;
    // uv_barrier_wait(&m_run_barrier); // kick off all threads in thread_main
    spin_barrier_wait(&m_run_barrier); // kick off all threads in thread_main
}

void ThreadPool::wait() {
    POWERSERVE_ASSERT(m_current_task != nullptr);
    // uv_barrier_wait(&m_run_barrier);
    spin_barrier_wait(&m_run_barrier);
    m_current_task = nullptr;
}

void ThreadPool::thread_main(size_t thread_id) {
    while (!m_exited) {
        // uv_barrier_wait(&m_run_barrier); // when main thread runs, it will wait for all threads to reach this barrier
        spin_barrier_wait(&m_run_barrier); // when main thread runs, it will wait for all threads to reach this barrier

        if (m_current_task) {
            m_current_task(thread_id);
            // uv_barrier_wait(&m_run_barrier);
            spin_barrier_wait(&m_run_barrier);
        }
    }
}

} // namespace powerserve
