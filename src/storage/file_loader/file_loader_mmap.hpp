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
#include "storage/file_loader.hpp"

#include <filesystem>
#include <span>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace powerserve::storage {

class FileLoaderMMap final : public FileLoader {
private:
    std::span<std::byte> m_mmap_space;

public:
    FileLoaderMMap(const std::filesystem::path &file_path) : FileLoader(file_path) {
        {
            const int fd = open(m_file_path.c_str(), O_RDONLY);
            m_file_handle.reset(fd);
        }

        struct stat file_stat;
        const int ret = fstat(m_file_handle.m_fd, &file_stat);
        if (ret != 0) [[unlikely]] {
            throw EnvironmentException("FileLoaderMMap", fmt::format("failed to fstat file {}", m_file_path));
        }

        const size_t file_size = file_stat.st_size;
        void *mmap_ret         = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, m_file_handle.m_fd, 0);
        if (mmap_ret == MAP_FAILED) [[unlikely]] {
            throw EnvironmentException("FileLoaderMMap", fmt::format("failed to mmap file {}", m_file_path));
        }

        m_mmap_space = {static_cast<std::byte *>(mmap_ret), file_size};
    }

    ~FileLoaderMMap() noexcept override {
        if (!m_mmap_space.empty()) {
            munmap(m_mmap_space.data(), m_mmap_space.size());
        }
    }

    FileLoaderMMap(const FileLoaderMMap &other) = delete;

    FileLoaderMMap(FileLoaderMMap &&other) noexcept : FileLoader(std::move(other)), m_mmap_space(other.m_mmap_space) {
        other.m_mmap_space = {};
    }

    FileLoaderMMap &operator=(const FileLoaderMMap &other) = delete;

    FileLoaderMMap &operator=(FileLoaderMMap &&other) noexcept {
        if (this != &other) {
            FileLoader::operator=(std::move(other));

            if (!m_mmap_space.empty()) {
                munmap(m_mmap_space.data(), m_mmap_space.size());
                other.m_mmap_space = {};
            }
            std::swap(m_mmap_space, other.m_mmap_space);
        }
        return *this;
    }

public:
    void load() override {
        /*
         * Prefetch the whole file into Page Cache
         */
        madvise(m_mmap_space.data(), m_mmap_space.size(), MADV_WILLNEED);
    }

    void unload() override {
        /*
         * Unload the whole file from Page Cache
         */
        madvise(m_mmap_space.data(), m_mmap_space.size(), MADV_DONTNEED);
    }

    std::span<std::byte> get_buffer(const bool implicit_load = true) override {
        if (implicit_load) {
            load();
        }
        return m_mmap_space;
    }

    FileLoaderMethod get_method() const override {
        return FileLoaderMethod::MMap;
    }
};

} // namespace powerserve::storage
