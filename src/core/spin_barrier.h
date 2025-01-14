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

#if defined(__cplusplus)
extern "C" {
#endif

#include <stddef.h>

struct spin_barrier {
    size_t opaque[2];
};

void spin_barrier_init(struct spin_barrier *opaque, size_t width);
void spin_barrier_wait(struct spin_barrier *opaque);

#if defined(__cplusplus)
}
#endif
