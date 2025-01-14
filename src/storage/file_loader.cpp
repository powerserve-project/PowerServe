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

#include "file_loader.hpp"

#include "file_loader/file_loader_bio.hpp"
#include "file_loader/file_loader_dio.hpp"
#include "file_loader/file_loader_mmap.hpp"
#include "file_loader/file_loader_uv.hpp"
#include "fmt/core.h"

#include <memory>
#include <unistd.h>

namespace powerserve::storage {

std::unique_ptr<FileLoader> build_file_loader(const std::filesystem::path &file_path, FileLoaderMethod method) {
    switch (method) {
    case FileLoaderMethod::BIO:
        return std::make_unique<FileLoaderBIO>(file_path);

    case FileLoaderMethod::MMap:
        return std::make_unique<FileLoaderMMap>(file_path);

    case FileLoaderMethod::DIO:
        return std::make_unique<FileLoaderDIO>(file_path);

    case FileLoaderMethod::UV:
        return std::make_unique<FileLoaderUV>(file_path);

    default:
        POWERSERVE_LOG_WARN("unknown file loader method {}, fallback to bio", static_cast<int>(method));
        return std::make_unique<FileLoaderBIO>(file_path);
    }
}

} // namespace powerserve::storage
