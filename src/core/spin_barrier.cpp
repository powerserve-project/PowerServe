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

#include "core/spin_barrier.hpp"
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <emmintrin.h>
#elif defined(__arm__) || defined(__aarch64__) || defined(_M_ARM64)
#include <arm_acle.h>
#endif

namespace powerserve {

void cpu_yield() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_pause();
#elif defined(__arm__) || defined(__aarch64__) || defined(_M_ARM64)
    __yield();
#else
    asm volatile("nop" ::: "memory");
#endif
}

void SpinBarrier::init(size_t new_width) {
    width = new_width;
    count = 0;
}

void SpinBarrier::wait() {
    size_t current = count.fetch_add(1, std::memory_order_acq_rel);
    size_t target  = (current / width + 1) * width;
    while (count.load(std::memory_order_relaxed) < target) {
        cpu_yield();
    }
}

} // namespace powerserve

extern "C" {

void spin_barrier_init(struct spin_barrier *opaque, size_t width) {
    auto *self = reinterpret_cast<powerserve::SpinBarrier *>(opaque);
    self->init(width);
}

void spin_barrier_wait(struct spin_barrier *opaque) {
    auto *self = reinterpret_cast<powerserve::SpinBarrier *>(opaque);
    self->wait();
}
}
