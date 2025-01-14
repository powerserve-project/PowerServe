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

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>

#ifdef __GNUC__
#ifdef __MINGW32__
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define LLAMA_ATTRIBUTE_FORMAT(...)
#endif

enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_FALCON,
    LLM_ARCH_BAICHUAN,
    LLM_ARCH_GROK,
    LLM_ARCH_GPT2,
    LLM_ARCH_GPTJ,
    LLM_ARCH_GPTNEOX,
    LLM_ARCH_MPT,
    LLM_ARCH_STARCODER,
    LLM_ARCH_REFACT,
    LLM_ARCH_BERT,
    LLM_ARCH_NOMIC_BERT,
    LLM_ARCH_JINA_BERT_V2,
    LLM_ARCH_BLOOM,
    LLM_ARCH_STABLELM,
    LLM_ARCH_QWEN,
    LLM_ARCH_QWEN2,
    LLM_ARCH_QWEN2MOE,
    LLM_ARCH_PHI2,
    LLM_ARCH_PHI3,
    LLM_ARCH_PLAMO,
    LLM_ARCH_CODESHELL,
    LLM_ARCH_ORION,
    LLM_ARCH_INTERNLM2,
    LLM_ARCH_MINICPM,
    LLM_ARCH_MINICPM3,
    LLM_ARCH_GEMMA,
    LLM_ARCH_GEMMA2,
    LLM_ARCH_STARCODER2,
    LLM_ARCH_MAMBA,
    LLM_ARCH_XVERSE,
    LLM_ARCH_COMMAND_R,
    LLM_ARCH_DBRX,
    LLM_ARCH_OLMO,
    LLM_ARCH_OLMOE,
    LLM_ARCH_OPENELM,
    LLM_ARCH_ARCTIC,
    LLM_ARCH_DEEPSEEK2,
    LLM_ARCH_CHATGLM,
    LLM_ARCH_BITNET,
    LLM_ARCH_T5,
    LLM_ARCH_T5ENCODER,
    LLM_ARCH_JAIS,
    LLM_ARCH_NEMOTRON,
    LLM_ARCH_EXAONE,
    LLM_ARCH_RWKV6,
    LLM_ARCH_GRANITE,
    LLM_ARCH_UNKNOWN,
};

using llama_token = int32_t;

enum llama_token_attr {
    LLAMA_TOKEN_ATTR_UNDEFINED    = 0,
    LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0,
    LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1,
    LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2,
    LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3,  // SPECIAL?
    LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4,
    LLAMA_TOKEN_ATTR_BYTE         = 1 << 5,
    LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6,
    LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7,
    LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8,
    LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
};

enum llama_vocab_type {
    LLAMA_VOCAB_TYPE_NONE = 0, // For models without vocab
    LLAMA_VOCAB_TYPE_SPM  = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
    LLAMA_VOCAB_TYPE_BPE  = 2, // GPT-2 tokenizer based on byte-level BPE
    LLAMA_VOCAB_TYPE_WPM  = 3, // BERT tokenizer based on WordPiece
    LLAMA_VOCAB_TYPE_UGM  = 4, // T5 tokenizer based on Unigram
    LLAMA_VOCAB_TYPE_RWKV = 5, // RWKV tokenizer based on greedy tokenization
};

enum llama_vocab_pre_type {
    LLAMA_VOCAB_PRE_TYPE_DEFAULT        = 0,
    LLAMA_VOCAB_PRE_TYPE_LLAMA3         = 1,
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM   = 2,
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER = 3,
    LLAMA_VOCAB_PRE_TYPE_FALCON         = 4,
    LLAMA_VOCAB_PRE_TYPE_MPT            = 5,
    LLAMA_VOCAB_PRE_TYPE_STARCODER      = 6,
    LLAMA_VOCAB_PRE_TYPE_GPT2           = 7,
    LLAMA_VOCAB_PRE_TYPE_REFACT         = 8,
    LLAMA_VOCAB_PRE_TYPE_COMMAND_R      = 9,
    LLAMA_VOCAB_PRE_TYPE_STABLELM2      = 10,
    LLAMA_VOCAB_PRE_TYPE_QWEN2          = 11,
    LLAMA_VOCAB_PRE_TYPE_OLMO           = 12,
    LLAMA_VOCAB_PRE_TYPE_DBRX           = 13,
    LLAMA_VOCAB_PRE_TYPE_SMAUG          = 14,
    LLAMA_VOCAB_PRE_TYPE_PORO           = 15,
    LLAMA_VOCAB_PRE_TYPE_CHATGLM3       = 16,
    LLAMA_VOCAB_PRE_TYPE_CHATGLM4       = 17,
    LLAMA_VOCAB_PRE_TYPE_VIKING         = 18,
    LLAMA_VOCAB_PRE_TYPE_JAIS           = 19,
    LLAMA_VOCAB_PRE_TYPE_TEKKEN         = 20,
    LLAMA_VOCAB_PRE_TYPE_SMOLLM         = 21,
    LLAMA_VOCAB_PRE_TYPE_CODESHELL      = 22,
    LLAMA_VOCAB_PRE_TYPE_BLOOM          = 23,
    LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH   = 24,
    LLAMA_VOCAB_PRE_TYPE_EXAONE         = 25,
};

enum llama_token_type { //TODO: remove, required until per token attributes are available from GGUF file
    LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
    LLAMA_TOKEN_TYPE_NORMAL       = 1,
    LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
    LLAMA_TOKEN_TYPE_CONTROL      = 3,
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
    LLAMA_TOKEN_TYPE_UNUSED       = 5,
    LLAMA_TOKEN_TYPE_BYTE         = 6,
};

struct llama_special_tokens {
    int32_t bos;
    int32_t eos;
    int32_t unk;
    int32_t sep;
    int32_t pad;
    int32_t cls;
    int32_t mask;
    int32_t prefix;
    int32_t suffix;
    int32_t middle;
    int32_t eot;
    int32_t eom;
};

struct llama_tokenizer_config {
    bool add_space_prefix;
    bool remove_extra_whitespaces;
    bool add_bos;
    bool add_eos;
};

struct llama_vocab {
    using id    = llama_token;
    using token = std::string;
    using tattr = llama_token_attr;

    struct token_data {
        token text;
        float score;
        tattr attr;
    };

    uint32_t n_vocab = 0; // TODO: not great because has to keep in sync with hparams.n_vocab

    enum llama_vocab_type     type     = LLAMA_VOCAB_TYPE_SPM;
    enum llama_vocab_pre_type type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;

    int max_token_len = 0; // used for optimizing longest token search

    std::unordered_map<token, id> token_to_id;
    std::vector<token_data>       id_to_token;

    std::vector<id>    cache_special_tokens;
    std::vector<token> cache_token_to_piece; // llama_token_to_piece(special = true);

    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    // default LLaMA special tokens
    id special_bos_id  = 1;
    id special_eos_id  = 2;
    id special_unk_id  = 0;
    id special_sep_id  = -1;
    id special_pad_id  = -1;
    id special_cls_id  = -1;
    id special_mask_id = -1;

    id linefeed_id       = 13;
    id special_prefix_id = -1;
    id special_suffix_id = -1;
    id special_middle_id = -1;
    id special_eot_id    = -1; // TODO: move above after "eos_id", and here add "file separator" token
    id special_eom_id    = -1;

    // tokenizer flags
    bool tokenizer_add_space_prefix = false;
    bool tokenizer_add_bos          = false;
    bool tokenizer_add_eos          = false;
    bool tokenizer_ignore_merges    = false;
    bool tokenizer_clean_spaces     = false;  // clean_up_tokenization_spaces
    bool tokenizer_remove_extra_whitespaces   = false;
    bool tokenizer_escape_whitespaces         = true;
    bool tokenizer_treat_whitespace_as_suffix = false;

    std::vector<char> precompiled_charsmap;

    int find_bpe_rank(const std::string & token_left, const std::string & token_right) const;
};

//
// internal API
//

// TODO: rename to llama_tokenize_impl
// TODO: This should probably be in llama.h
std::vector<llama_vocab::id> llama_tokenize_internal(
        const llama_vocab & vocab,
        std::string raw_text,
        bool add_special,
        bool parse_special = false);

// TODO: move the API below as member functions of llama_vocab
llama_token llama_byte_to_token_impl(const llama_vocab & vocab, uint8_t ch);

const char * llama_token_get_text_impl(const struct llama_vocab & vocab, llama_token token);

float llama_token_get_score_impl(const struct llama_vocab & vocab, llama_token token);

llama_token_attr llama_token_get_attr_impl(const struct llama_vocab & vocab, llama_token token);

bool llama_token_is_eog_impl(const struct llama_vocab & vocab, llama_token token);

bool llama_token_is_control_impl(const struct llama_vocab & vocab, llama_token token);

llama_token llama_token_bos_impl(const struct llama_vocab & vocab);
llama_token llama_token_eos_impl(const struct llama_vocab & vocab);
llama_token llama_token_cls_impl(const struct llama_vocab & vocab);
llama_token llama_token_sep_impl(const struct llama_vocab & vocab);
llama_token llama_token_nl_impl (const struct llama_vocab & vocab);
llama_token llama_token_pad_impl(const struct llama_vocab & vocab);

bool llama_add_bos_token_impl(const struct llama_vocab & vocab);
bool llama_add_eos_token_impl(const struct llama_vocab & vocab);

llama_token llama_token_prefix_impl(const struct llama_vocab & vocab);
llama_token llama_token_middle_impl(const struct llama_vocab & vocab);
llama_token llama_token_suffix_impl(const struct llama_vocab & vocab);
llama_token llama_token_eot_impl   (const struct llama_vocab & vocab);
llama_token llama_token_eom_impl   (const struct llama_vocab & vocab);

int32_t llama_tokenize_impl(
        const struct llama_vocab & vocab,
                      const char * text,
                         int32_t   text_len,
                     llama_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);

// does not write null-terminator to buf
int32_t llama_token_to_piece_impl(
        const struct llama_vocab & vocab,
                     llama_token   token,
                            char * buf,
                         int32_t   length,
                         int32_t   lstrip,
                            bool   special);

int32_t llama_detokenize_impl(
        const struct llama_vocab & vocab,
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special);

void llm_load_vocab(llama_vocab & vocab, struct gguf_context * ctx);
std::string llama_token_to_piece(const struct llama_vocab & vocab, llama_token token, bool special = true);
