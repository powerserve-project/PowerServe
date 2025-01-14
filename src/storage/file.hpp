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

#include "core/exception.hpp"

#include <unistd.h>

namespace powerserve::storage {

///
/// RAII Wrapper for file descriptor
///
struct FileHandle {
public:
    static constexpr int INVALID_FILE_HANDLE = -1;

public:
    /// File descriptor
    int m_fd = INVALID_FILE_HANDLE;

public:
    FileHandle() = default;

    FileHandle(const int fd) : m_fd(fd) {}

    ~FileHandle() noexcept {
        reset();
    }

    FileHandle(const FileHandle &other) = delete;

    FileHandle(FileHandle &&other) noexcept : m_fd(other.m_fd) {
        other.m_fd = INVALID_FILE_HANDLE;
    }

    FileHandle &operator=(const FileHandle &other) = delete;

    FileHandle &operator=(FileHandle &&other) noexcept {
        if (this != &other) {
            reset(other.m_fd);
            other.m_fd = INVALID_FILE_HANDLE;
        }
        return *this;
    }

public:
    /*!
     * @brief Reset file descriptor with a new one
     * @param[in] new_fd New file descriptor assigned to the handle
     * @note If the file handle has already been assigned, it will close the old one and hold the new one.
     */
    void reset(const int new_fd = INVALID_FILE_HANDLE) {
        if (m_fd != INVALID_FILE_HANDLE) {
            const int ret = close(m_fd);
            if (ret == -1) [[unlikely]] {
                throw EnvironmentException("FileHandle", fmt::format("failed to close file {}", m_fd));
            }
        }
        m_fd = new_fd;
    }

    /*!
     * @brief Reset file descriptor directly
     * @note This may lead to resources leak. Only use it after handling the file descriptor properly
     */
    void unsafe_reset() {
        m_fd = INVALID_FILE_HANDLE;
    }
};

} // namespace powerserve::storage
