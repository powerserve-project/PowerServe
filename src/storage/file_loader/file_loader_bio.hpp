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

#include <cstddef>
#include <filesystem>
#include <span>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace powerserve::storage {

class FileLoaderBIO final : public FileLoader {
private:
    std::span<std::byte> m_buffer;

public:
    FileLoaderBIO(const std::filesystem::path &file_path) : FileLoader(file_path) {
        const int fd = open(m_file_path.c_str(), O_RDONLY);
        m_file_handle.reset(fd);
    }

    ~FileLoaderBIO() noexcept override {
        unload();
    }

    FileLoaderBIO(const FileLoaderBIO &other) = delete;

    FileLoaderBIO(FileLoaderBIO &&other) noexcept : FileLoader(std::move(other)), m_buffer(other.m_buffer) {
        other.m_buffer = {};
    }

    FileLoaderBIO &operator=(const FileLoaderBIO &other) = delete;

    FileLoaderBIO &operator=(FileLoaderBIO &&other) noexcept {
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

        struct stat file_stat;
        {
            const int ret = fstat(m_file_handle.m_fd, &file_stat);
            if (ret != 0) [[unlikely]] {
                throw EnvironmentException("FileLoaderBIO", fmt::format("failed to fstat file {}", m_file_path));
            }
        }

        /*
         * Allocate buffer
         */
        const size_t file_size = file_stat.st_size;
        std::byte *buffer_ptr  = new std::byte[file_size];
        m_buffer               = {buffer_ptr, file_size};

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
        return FileLoaderMethod::BIO;
    }
};

} // namespace powerserve::storage
