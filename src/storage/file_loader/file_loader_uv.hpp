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
#include "core/logger.hpp"
#include "storage/file_loader.hpp"
#include "uv.h"

#include <cstddef>
#include <span>

namespace powerserve::storage {

class FileLoaderUV final : public FileLoader {
private:
    uv_fs_t m_request;

    std::span<std::byte> m_buffer;

public:
    FileLoaderUV(const std::filesystem::path &file_path) : FileLoader(file_path) {
        uv_fs_open(nullptr, &m_request, file_path.c_str(), O_RDONLY, 0, nullptr);
        const uv_file fd = static_cast<uv_file>(m_request.result);
        if (fd < 0) [[unlikely]] {
            throw EnvironmentException("FileLoaderUV", fmt::format("failed to open file: {}", file_path));
        }
        uv_fs_req_cleanup(&m_request);
        m_file_handle.reset(fd);
    }

    ~FileLoaderUV() noexcept override {
        unload();

        uv_fs_close(nullptr, &m_request, m_file_handle.m_fd, nullptr);
        uv_fs_req_cleanup(&m_request);
        m_file_handle.unsafe_reset(); // Avoid redundant closing
    }

    FileLoaderUV(const FileLoaderUV &other) = delete;

    FileLoaderUV(FileLoaderUV &&other) noexcept : FileLoader(std::move(other)), m_buffer(other.m_buffer) {
        other.m_buffer = {};
    }

    FileLoaderUV &operator=(const FileLoaderUV &other) = delete;

    FileLoaderUV &operator=(FileLoaderUV &&other) noexcept {
        if (this != &other) {
            FileLoader::operator=(std::move(other));

            unload();
            m_buffer = other.m_buffer;
        }
        return *this;
    }

public:
    void load() override {
        if (!m_buffer.empty()) {
            POWERSERVE_LOG_WARN("trying to load a buffer twice");
            unload();
        }

        uv_fs_fstat(nullptr, &m_request, m_file_handle.m_fd, nullptr);
        if (m_request.result != 0) [[unlikely]] {
            throw EnvironmentException("FileLoaderUV", fmt::format("failed to fstat file {}", m_file_path));
        }

        const size_t file_size = m_request.statbuf.st_size;
        uv_fs_req_cleanup(&m_request);

        /*
         * Allocate buffer
         */
        std::byte *buffer_ptr = new std::byte[file_size];
        m_buffer              = {buffer_ptr, file_size};

        uv_buf_t buf = {
            .base = (char *)buffer_ptr,
            .len  = file_size,
        };
        uv_fs_read(nullptr, &m_request, m_file_handle.m_fd, &buf, 1, 0, nullptr);
        if (m_request.result != static_cast<ssize_t>(file_size)) [[unlikely]] {
            throw EnvironmentException("FileLoaderUV", fmt::format("failed to fstat file {}", m_file_path));
        }
        uv_fs_req_cleanup(&m_request);

        /*
         * Read the whole file into the buffer
         */
        {
            const ssize_t ret = pread(m_file_handle.m_fd, buffer_ptr, file_size, 0);
            if (ret != static_cast<ssize_t>(file_size)) [[unlikely]] {
                throw EnvironmentException(
                    "FileLoaderUV",
                    fmt::format("failed to read {} bytes from file {} (ret = {})", file_size, m_file_path, ret)
                );
            }
        }
    }

    void unload() override {
        if (!m_buffer.empty()) {
            delete[] m_buffer.data();
        }
        m_buffer = {};
    }

    std::span<std::byte> get_buffer(const bool implicit_load = true) override {
        if (implicit_load) {
            load();
        }
        return m_buffer;
    }

    FileLoaderMethod get_method() const override {
        return FileLoaderMethod::UV;
    }
};

} // namespace powerserve::storage
