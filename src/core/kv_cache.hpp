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

#include "core/defines.hpp"
#include "core/exception.hpp"
#include "core/logger.hpp"

namespace powerserve {

#define POWERSERVE_ASSERT_KVCACHE(expr, ...) POWERSERVE_ASSERT_MODULE(expr, "Core", "KVCache", __VA_ARGS__)

struct KVPosition {
    size_t layer_id = 0;
    size_t head_id  = 0;
    size_t index    = 0;
};

struct KVView {
    size_t n_elements   = 0;
    size_t element_size = 0;
    size_t stride       = 0;
    void *data          = nullptr;

    bool is_contiguous() const {
        return element_size == stride;
    }

    ALWAYS_INLINE void copy_from(KVView other) {
        POWERSERVE_ASSERT_KVCACHE(n_elements == other.n_elements);
        POWERSERVE_ASSERT_KVCACHE(element_size == other.element_size);

        if (POWERSERVE_LIKELY(is_contiguous() && other.is_contiguous())) {
            memcpy(data, other.data, n_elements * element_size);
        } else if (other.is_contiguous() && element_size == 2) {
            auto src = (uint16_t *)other.data;
            auto dst = (uint8_t *)data;

            // clang-format off
            #pragma unroll(4)
            // clang-format on
            for (size_t i = 0; i < n_elements; i++) {
                *(uint16_t *)dst = *src;
                src++;
                dst += stride;
            }
        } else if (element_size == 2) {
            auto src = (uint8_t *)other.data;
            auto dst = (uint8_t *)data;

            // clang-format off
            #pragma unroll(4)
            // clang-format on
            for (size_t i = 0; i < n_elements; i++) {
                *(uint16_t *)dst = *(uint16_t *)src;
                src += other.stride;
                dst += stride;
            }
        } else if (element_size == 4) {
            auto src = (uint8_t *)other.data;
            auto dst = (uint8_t *)data;

            // clang-format off
            #pragma unroll(4)
            // clang-format on
            for (size_t i = 0; i < n_elements; i++) {
                *(uint32_t *)dst = *(uint32_t *)src;
                src += other.stride;
                dst += stride;
            }
        } else {
            POWERSERVE_ABORT("unsupported layout of KV");
        }
    }
};

struct KVInterfaceBase {
    auto get_key(KVPosition token_pos) const -> KVView;
    auto get_value(KVPosition token_pos) const -> KVView;
    auto key_entry(KVPosition cache_pos) const -> KVView;
    auto value_entry(KVPosition cache_pos) const -> KVView;
    void set_mask(size_t cache_index, bool mask);
};

struct KVCacheInterface {
    const size_t n_layers;
    const size_t n_heads;
    const size_t size;
    size_t position = 0;

    KVCacheInterface(size_t n_layers, size_t n_heads, size_t size) : n_layers(n_layers), n_heads(n_heads), size(size) {}

    virtual ~KVCacheInterface() = default;

    // Get view of a key data in a batch at token_pos
    virtual auto key_data(KVPosition token_pos) const -> KVView = 0;

    // Get view of a value data in a batch at token_pos
    virtual auto value_data(KVPosition token_pos) const -> KVView = 0;

    // Get view of a key cache entry at cache_pos
    virtual auto key_entry(KVPosition cache_pos) const -> KVView = 0;

    // Get view of a value cache entry at cache_pos
    virtual auto value_entry(KVPosition cache_pos) const -> KVView = 0;

    // Copy a token's kv at src_token_index to kv cache entry at dst_cache_index
    // For layers in the range [start_layer_id, end_layer_id)
    virtual void copy_for_layers(
        size_t start_layer_id, size_t end_layer_id, size_t dst_cache_index, size_t src_token_index
    ) = 0;

    void copy(size_t dst_cache_index, size_t src_token_index) {
        copy_for_layers(0, n_layers, dst_cache_index, src_token_index);
    }

    // Copy kv cache entry at src_cache_index to kv cache entry at dst_cache_index
    virtual void move(size_t dst_cache_index, size_t src_cache_index) = 0;

    // Mask kv cache entry at cache_index
    virtual void mask(size_t cache_index) = 0;

    // Unmask kv cache entry at cache_index
    virtual void unmask(size_t cache_index) = 0;

    // Copy n_tokens token kv to the end of cache after position. Position is not modified
    // For layers in the range [start_layer_id, end_layer_id)
    virtual void save_tokens_for_layers(size_t start_layer_id, size_t end_layer_id, size_t n_tokens) = 0;

    void save_tokens(size_t n_tokens) {
        save_tokens_for_layers(0, n_layers, n_tokens);
    }

    // Unmask n_tokens tokens after position. Position is not modified
    virtual void unmask_tokens(size_t n_tokens) = 0;

    // Unmask tokens and increment position by n_tokens. Old position is returned
    virtual size_t advance_tokens(size_t n_tokens) = 0;

    // Mask tokens and decrement position by n_tokens. Old position is returned
    virtual size_t rollback_tokens(size_t n_tokens) = 0;

    // Truncate cache to n_tokens. Old position is returned
    virtual size_t truncate_tokens(size_t n_tokens) = 0;

    size_t append_tokens(size_t n_tokens) {
        save_tokens(n_tokens);
        unmask_tokens(n_tokens);
        return advance_tokens(n_tokens);
    }
};

template <typename KVInterface>
struct KVCache final : KVCacheInterface {
    template <typename... Args>
    KVCache(size_t n_layers, size_t n_heads, size_t size, Args &&...args) :
        KVCacheInterface(n_layers, n_heads, size),
        interface(std::forward<Args>(args)...) {}

    auto key_data(KVPosition token_pos) const -> KVView override {
        return interface.get_key(token_pos);
    }

    auto value_data(KVPosition token_pos) const -> KVView override {
        return interface.get_value(token_pos);
    }

    auto key_entry(KVPosition cache_pos) const -> KVView override {
        return interface.key_entry(cache_pos);
    }

    auto value_entry(KVPosition cache_pos) const -> KVView override {
        return interface.value_entry(cache_pos);
    }

    void copy_for_layers(size_t start_layer_id, size_t end_layer_id, size_t dst_cache_index, size_t src_token_index)
        override {
        for (size_t i = start_layer_id; i < end_layer_id; i++) {
            for (size_t j = 0; j < n_heads; j++) {
                auto src_key = interface.get_key({.layer_id = i, .head_id = j, .index = src_token_index});
                auto dst_key = interface.key_entry({.layer_id = i, .head_id = j, .index = dst_cache_index});
                dst_key.copy_from(src_key);

                auto src_value = interface.get_value({.layer_id = i, .head_id = j, .index = src_token_index});
                auto dst_value = interface.value_entry({.layer_id = i, .head_id = j, .index = dst_cache_index});
                dst_value.copy_from(src_value);
            }
        }
    }

    void move(size_t dst_cache_index, size_t src_cache_index) override {
        if (dst_cache_index == src_cache_index) {
            return;
        }

        for (size_t i = 0; i < n_layers; i++) {
            for (size_t j = 0; j < n_heads; j++) {
                auto src_key = interface.key_entry({.layer_id = i, .head_id = j, .index = src_cache_index});
                auto dst_key = interface.key_entry({.layer_id = i, .head_id = j, .index = dst_cache_index});
                dst_key.copy_from(src_key);

                auto src_value = interface.value_entry({.layer_id = i, .head_id = j, .index = src_cache_index});
                auto dst_value = interface.value_entry({.layer_id = i, .head_id = j, .index = dst_cache_index});
                dst_value.copy_from(src_value);
            }
        }
    }

    void mask(size_t cache_index) override {
        POWERSERVE_ASSERT_KVCACHE(cache_index < position);
        interface.set_mask(cache_index, true);
    }

    void unmask(size_t cache_index) override {
        POWERSERVE_ASSERT_KVCACHE(cache_index < position);
        interface.set_mask(cache_index, false);
    }

    void save_tokens_for_layers(size_t start_layer_id, size_t end_layer_id, size_t n_tokens) override {
        POWERSERVE_ASSERT_KVCACHE(
            position + n_tokens <= size, "the length of kvcache is up to the preset threshold: {}", size
        );
        for (size_t i = 0; i < n_tokens; i++) {
            copy_for_layers(start_layer_id, end_layer_id, position + i, i);
        }
    }

    void unmask_tokens(size_t n_tokens) override {
        POWERSERVE_ASSERT_KVCACHE(
            position + n_tokens <= size, "the length of kvcache is up to the preset threshold: {}", size
        );
        for (size_t i = 0; i < n_tokens; i++) {
            interface.set_mask(position + i, false);
        }
    }

    size_t advance_tokens(size_t n_tokens) override {
        unmask_tokens(n_tokens);
        size_t old_position = position;
        position += n_tokens;
        return old_position;
    }

    size_t rollback_tokens(size_t n_tokens) override {
        POWERSERVE_ASSERT_KVCACHE(position >= n_tokens);
        size_t old_position = position;
        position -= n_tokens;
        for (size_t i = 0; i < n_tokens; i++) {
            interface.set_mask(position + i, true);
        }
        return old_position;
    }

    size_t truncate_tokens(size_t n_tokens) override {
        size_t old_position = position;
        if (n_tokens < position) {
            rollback_tokens(position - n_tokens);
        }
        return old_position;
    }

private:
    KVInterface interface;
};

} // namespace powerserve
