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

#define POWERSERVE_BUILTIN_EXPECT(expr, value) __builtin_expect((expr), (value))
#define POWERSERVE_LIKELY(expr)                POWERSERVE_BUILTIN_EXPECT((expr), 1)
#define POWERSERVE_UNLIKELY(expr)              POWERSERVE_BUILTIN_EXPECT((expr), 0)

#if !defined(ALWAYS_INLINE)
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

#define POWERSERVE_UNUSED(x) ((void)(x))
