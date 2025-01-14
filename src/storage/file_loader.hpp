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

#include "file.hpp"

#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <span>

namespace powerserve::storage {

enum class FileLoaderMethod : int {
    /// Load file using mmap
    MMap = 0,
    /// Load file using buffered I/O interface
    BIO = 1,
    /// Load file using direct I/O interface
    DIO = 2,
    /// Load file using libuv I/O interface
    UV = 3
};

///
/// Simple loader for file-granularity Reading
/// @note Only reading is supported. Writing file buffer leads to undefined behaviours
///
class FileLoader {
protected:
    /// The path to the file
    std::filesystem::path m_file_path;
    /// The file descriptor
    FileHandle m_file_handle;

public:
    FileLoader(const std::filesystem::path &file_path) : m_file_path(file_path) {}

    virtual ~FileLoader() noexcept = default;

    FileLoader(const FileLoader &other) = delete;

    FileLoader(FileLoader &&other) noexcept = default;

    FileLoader &operator=(const FileLoader &other) = delete;

    FileLoader &operator=(FileLoader &&other) noexcept = default;

public: /* Buffer Operation */
    /*!
     * @brief Load the whole file into buffer
     * @note This may incurs large memory allocation
     */
    virtual void load() = 0;

    /*!
     * @brief Release file buffer while keeping the file handle
     */
    virtual void unload() = 0;

public: /* Getter */
    /*!
     * @brief Get the file buffer
     * @param[in] implicit_load Read the file into the buffer if it hasn't been loaded.
     * @note Getting a buffer without pre-load operation or implicit load flag leads to undefined behaviour.
     */
    virtual std::span<std::byte> get_buffer(bool implicit_load = true) = 0;

    template <class T>
    inline std::span<T> get_buffer(bool implicit_load = true) {
        std::span<std::byte> origin_buffer = get_buffer(implicit_load);
        T *buffer_ptr                      = reinterpret_cast<T *>(origin_buffer.data());
        const size_t buffer_size           = origin_buffer.size();
        return {buffer_ptr, buffer_size / sizeof(T)};
    }

    /*!
     * @brief Get the underlying implementation method of FileLoader
     */
    virtual FileLoaderMethod get_method() const = 0;
};

/*!
 * @brief Factory Function: Build up a file loader
 */
std::unique_ptr<FileLoader> build_file_loader(const std::filesystem::path &file_path, FileLoaderMethod method);

} // namespace powerserve::storage
