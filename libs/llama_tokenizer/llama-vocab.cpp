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

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cstdarg>
#include <cstring>
#include <forward_list>
#include <queue>
#include <sstream>

#include "llama-vocab.h"
#include "unicode.h"
#include "ggml.h"

//
// helpers
//

static void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

LLAMA_ATTRIBUTE_FORMAT(1, 2)
static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    assert(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    assert(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_LLAMA,           "llama"        },
    { LLM_ARCH_FALCON,          "falcon"       },
    { LLM_ARCH_GROK,            "grok"         },
    { LLM_ARCH_GPT2,            "gpt2"         },
    { LLM_ARCH_GPTJ,            "gptj"         },
    { LLM_ARCH_GPTNEOX,         "gptneox"      },
    { LLM_ARCH_MPT,             "mpt"          },
    { LLM_ARCH_BAICHUAN,        "baichuan"     },
    { LLM_ARCH_STARCODER,       "starcoder"    },
    { LLM_ARCH_REFACT,          "refact"       },
    { LLM_ARCH_BERT,            "bert"         },
    { LLM_ARCH_NOMIC_BERT,      "nomic-bert"   },
    { LLM_ARCH_JINA_BERT_V2,    "jina-bert-v2" },
    { LLM_ARCH_BLOOM,           "bloom"        },
    { LLM_ARCH_STABLELM,        "stablelm"     },
    { LLM_ARCH_QWEN,            "qwen"         },
    { LLM_ARCH_QWEN2,           "qwen2"        },
    { LLM_ARCH_QWEN2MOE,        "qwen2moe"     },
    { LLM_ARCH_PHI2,            "phi2"         },
    { LLM_ARCH_PHI3,            "phi3"         },
    { LLM_ARCH_PLAMO,           "plamo"        },
    { LLM_ARCH_CODESHELL,       "codeshell"    },
    { LLM_ARCH_ORION,           "orion"        },
    { LLM_ARCH_INTERNLM2,       "internlm2"    },
    { LLM_ARCH_MINICPM,         "minicpm"      },
    { LLM_ARCH_MINICPM3,        "minicpm3"     },
    { LLM_ARCH_GEMMA,           "gemma"        },
    { LLM_ARCH_GEMMA2,          "gemma2"       },
    { LLM_ARCH_STARCODER2,      "starcoder2"   },
    { LLM_ARCH_MAMBA,           "mamba"        },
    { LLM_ARCH_XVERSE,          "xverse"       },
    { LLM_ARCH_COMMAND_R,       "command-r"    },
    { LLM_ARCH_DBRX,            "dbrx"         },
    { LLM_ARCH_OLMO,            "olmo"         },
    { LLM_ARCH_OLMOE,           "olmoe"        },
    { LLM_ARCH_OPENELM,         "openelm"      },
    { LLM_ARCH_ARCTIC,          "arctic"       },
    { LLM_ARCH_DEEPSEEK2,       "deepseek2"    },
    { LLM_ARCH_CHATGLM,         "chatglm"      },
    { LLM_ARCH_BITNET,          "bitnet"       },
    { LLM_ARCH_T5,              "t5"           },
    { LLM_ARCH_T5ENCODER,       "t5encoder"    },
    { LLM_ARCH_JAIS,            "jais"         },
    { LLM_ARCH_NEMOTRON,        "nemotron"     },
    { LLM_ARCH_EXAONE,          "exaone"       },
    { LLM_ARCH_RWKV6,           "rwkv6"        },
    { LLM_ARCH_GRANITE,         "granite"      },
    { LLM_ARCH_UNKNOWN,         "(unknown)"    },
};

enum llm_kv {
    LLM_KV_GENERAL_TYPE,
    LLM_KV_GENERAL_ARCHITECTURE,
    LLM_KV_GENERAL_QUANTIZATION_VERSION,
    LLM_KV_GENERAL_ALIGNMENT,
    LLM_KV_GENERAL_NAME,
    LLM_KV_GENERAL_AUTHOR,
    LLM_KV_GENERAL_VERSION,
    LLM_KV_GENERAL_URL,
    LLM_KV_GENERAL_DESCRIPTION,
    LLM_KV_GENERAL_LICENSE,
    LLM_KV_GENERAL_SOURCE_URL,
    LLM_KV_GENERAL_SOURCE_HF_REPO,

    LLM_KV_VOCAB_SIZE,
    LLM_KV_CONTEXT_LENGTH,
    LLM_KV_EMBEDDING_LENGTH,
    LLM_KV_BLOCK_COUNT,
    LLM_KV_LEADING_DENSE_BLOCK_COUNT,
    LLM_KV_FEED_FORWARD_LENGTH,
    LLM_KV_EXPERT_FEED_FORWARD_LENGTH,
    LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH,
    LLM_KV_USE_PARALLEL_RESIDUAL,
    LLM_KV_TENSOR_DATA_LAYOUT,
    LLM_KV_EXPERT_COUNT,
    LLM_KV_EXPERT_USED_COUNT,
    LLM_KV_EXPERT_SHARED_COUNT,
    LLM_KV_EXPERT_WEIGHTS_SCALE,
    LLM_KV_POOLING_TYPE,
    LLM_KV_LOGIT_SCALE,
    LLM_KV_DECODER_START_TOKEN_ID,
    LLM_KV_ATTN_LOGIT_SOFTCAPPING,
    LLM_KV_FINAL_LOGIT_SOFTCAPPING,
    LLM_KV_RESCALE_EVERY_N_LAYERS,
    LLM_KV_TIME_MIX_EXTRA_DIM,
    LLM_KV_TIME_DECAY_EXTRA_DIM,
    LLM_KV_RESIDUAL_SCALE,
    LLM_KV_EMBEDDING_SCALE,

    LLM_KV_ATTENTION_HEAD_COUNT,
    LLM_KV_ATTENTION_HEAD_COUNT_KV,
    LLM_KV_ATTENTION_MAX_ALIBI_BIAS,
    LLM_KV_ATTENTION_CLAMP_KQV,
    LLM_KV_ATTENTION_KEY_LENGTH,
    LLM_KV_ATTENTION_VALUE_LENGTH,
    LLM_KV_ATTENTION_LAYERNORM_EPS,
    LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,
    LLM_KV_ATTENTION_CAUSAL,
    LLM_KV_ATTENTION_Q_LORA_RANK,
    LLM_KV_ATTENTION_KV_LORA_RANK,
    LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT,
    LLM_KV_ATTENTION_SLIDING_WINDOW,
    LLM_KV_ATTENTION_SCALE,

    LLM_KV_ROPE_DIMENSION_COUNT,
    LLM_KV_ROPE_FREQ_BASE,
    LLM_KV_ROPE_SCALE_LINEAR,
    LLM_KV_ROPE_SCALING_TYPE,
    LLM_KV_ROPE_SCALING_FACTOR,
    LLM_KV_ROPE_SCALING_ATTN_FACTOR,
    LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,
    LLM_KV_ROPE_SCALING_FINETUNED,
    LLM_KV_ROPE_SCALING_YARN_LOG_MUL,

    LLM_KV_SPLIT_NO,
    LLM_KV_SPLIT_COUNT,
    LLM_KV_SPLIT_TENSORS_COUNT,

    LLM_KV_SSM_INNER_SIZE,
    LLM_KV_SSM_CONV_KERNEL,
    LLM_KV_SSM_STATE_SIZE,
    LLM_KV_SSM_TIME_STEP_RANK,
    LLM_KV_SSM_DT_B_C_RMS,

    LLM_KV_WKV_HEAD_SIZE,

    LLM_KV_TOKENIZER_MODEL,
    LLM_KV_TOKENIZER_PRE,
    LLM_KV_TOKENIZER_LIST,
    LLM_KV_TOKENIZER_TOKEN_TYPE,
    LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,
    LLM_KV_TOKENIZER_SCORES,
    LLM_KV_TOKENIZER_MERGES,
    LLM_KV_TOKENIZER_BOS_ID,
    LLM_KV_TOKENIZER_EOS_ID,
    LLM_KV_TOKENIZER_UNK_ID,
    LLM_KV_TOKENIZER_SEP_ID,
    LLM_KV_TOKENIZER_PAD_ID,
    LLM_KV_TOKENIZER_CLS_ID,
    LLM_KV_TOKENIZER_MASK_ID,
    LLM_KV_TOKENIZER_ADD_BOS,
    LLM_KV_TOKENIZER_ADD_EOS,
    LLM_KV_TOKENIZER_ADD_PREFIX,
    LLM_KV_TOKENIZER_REMOVE_EXTRA_WS,
    LLM_KV_TOKENIZER_PRECOMPILED_CHARSMAP,
    LLM_KV_TOKENIZER_HF_JSON,
    LLM_KV_TOKENIZER_RWKV,
    LLM_KV_TOKENIZER_PREFIX_ID,
    LLM_KV_TOKENIZER_SUFFIX_ID,
    LLM_KV_TOKENIZER_MIDDLE_ID,
    LLM_KV_TOKENIZER_EOT_ID,
    LLM_KV_TOKENIZER_EOM_ID,

    LLM_KV_ADAPTER_TYPE,
    LLM_KV_ADAPTER_LORA_ALPHA,
};

static const std::map<llm_kv, const char *> LLM_KV_NAMES = {
    { LLM_KV_GENERAL_TYPE,                  "general.type"                          },
    { LLM_KV_GENERAL_ARCHITECTURE,          "general.architecture"                  },
    { LLM_KV_GENERAL_QUANTIZATION_VERSION,  "general.quantization_version"          },
    { LLM_KV_GENERAL_ALIGNMENT,             "general.alignment"                     },
    { LLM_KV_GENERAL_NAME,                  "general.name"                          },
    { LLM_KV_GENERAL_AUTHOR,                "general.author"                        },
    { LLM_KV_GENERAL_VERSION,               "general.version"                       },
    { LLM_KV_GENERAL_URL,                   "general.url"                           },
    { LLM_KV_GENERAL_DESCRIPTION,           "general.description"                   },
    { LLM_KV_GENERAL_LICENSE,               "general.license"                       },
    { LLM_KV_GENERAL_SOURCE_URL,            "general.source.url"                    },
    { LLM_KV_GENERAL_SOURCE_HF_REPO,        "general.source.huggingface.repository" },

    { LLM_KV_VOCAB_SIZE,                        "%s.vocab_size"                        },
    { LLM_KV_CONTEXT_LENGTH,                    "%s.context_length"                    },
    { LLM_KV_EMBEDDING_LENGTH,                  "%s.embedding_length"                  },
    { LLM_KV_BLOCK_COUNT,                       "%s.block_count"                       },
    { LLM_KV_LEADING_DENSE_BLOCK_COUNT,         "%s.leading_dense_block_count"         },
    { LLM_KV_FEED_FORWARD_LENGTH,               "%s.feed_forward_length"               },
    { LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        "%s.expert_feed_forward_length"        },
    { LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, "%s.expert_shared_feed_forward_length" },
    { LLM_KV_USE_PARALLEL_RESIDUAL,             "%s.use_parallel_residual"             },
    { LLM_KV_TENSOR_DATA_LAYOUT,                "%s.tensor_data_layout"                },
    { LLM_KV_EXPERT_COUNT,                      "%s.expert_count"                      },
    { LLM_KV_EXPERT_USED_COUNT,                 "%s.expert_used_count"                 },
    { LLM_KV_EXPERT_SHARED_COUNT,               "%s.expert_shared_count"               },
    { LLM_KV_EXPERT_WEIGHTS_SCALE,              "%s.expert_weights_scale"              },
    { LLM_KV_POOLING_TYPE,                      "%s.pooling_type"                      },
    { LLM_KV_LOGIT_SCALE,                       "%s.logit_scale"                       },
    { LLM_KV_DECODER_START_TOKEN_ID,            "%s.decoder_start_token_id"            },
    { LLM_KV_ATTN_LOGIT_SOFTCAPPING,            "%s.attn_logit_softcapping"            },
    { LLM_KV_FINAL_LOGIT_SOFTCAPPING,           "%s.final_logit_softcapping"           },
    { LLM_KV_RESCALE_EVERY_N_LAYERS,            "%s.rescale_every_n_layers"            },
    { LLM_KV_TIME_MIX_EXTRA_DIM,                "%s.time_mix_extra_dim"                },
    { LLM_KV_TIME_DECAY_EXTRA_DIM,              "%s.time_decay_extra_dim"              },
    { LLM_KV_RESIDUAL_SCALE,                    "%s.residual_scale"                    },
    { LLM_KV_EMBEDDING_SCALE,                   "%s.embedding_scale"                   },

    { LLM_KV_ATTENTION_HEAD_COUNT,             "%s.attention.head_count"             },
    { LLM_KV_ATTENTION_HEAD_COUNT_KV,          "%s.attention.head_count_kv"          },
    { LLM_KV_ATTENTION_MAX_ALIBI_BIAS,         "%s.attention.max_alibi_bias"         },
    { LLM_KV_ATTENTION_CLAMP_KQV,              "%s.attention.clamp_kqv"              },
    { LLM_KV_ATTENTION_KEY_LENGTH,             "%s.attention.key_length"             },
    { LLM_KV_ATTENTION_VALUE_LENGTH,           "%s.attention.value_length"           },
    { LLM_KV_ATTENTION_LAYERNORM_EPS,          "%s.attention.layer_norm_epsilon"     },
    { LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      "%s.attention.layer_norm_rms_epsilon" },
    { LLM_KV_ATTENTION_CAUSAL,                 "%s.attention.causal"                 },
    { LLM_KV_ATTENTION_Q_LORA_RANK,            "%s.attention.q_lora_rank"            },
    { LLM_KV_ATTENTION_KV_LORA_RANK,           "%s.attention.kv_lora_rank"           },
    { LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, "%s.attention.relative_buckets_count" },
    { LLM_KV_ATTENTION_SLIDING_WINDOW,         "%s.attention.sliding_window"         },
    { LLM_KV_ATTENTION_SCALE,                  "%s.attention.scale"                  },

    { LLM_KV_ROPE_DIMENSION_COUNT,          "%s.rope.dimension_count"                 },
    { LLM_KV_ROPE_FREQ_BASE,                "%s.rope.freq_base"                       },
    { LLM_KV_ROPE_SCALE_LINEAR,             "%s.rope.scale_linear"                    },
    { LLM_KV_ROPE_SCALING_TYPE,             "%s.rope.scaling.type"                    },
    { LLM_KV_ROPE_SCALING_FACTOR,           "%s.rope.scaling.factor"                  },
    { LLM_KV_ROPE_SCALING_ATTN_FACTOR,      "%s.rope.scaling.attn_factor"             },
    { LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,     "%s.rope.scaling.original_context_length" },
    { LLM_KV_ROPE_SCALING_FINETUNED,        "%s.rope.scaling.finetuned"               },
    { LLM_KV_ROPE_SCALING_YARN_LOG_MUL,     "%s.rope.scaling.yarn_log_multiplier"     },

    { LLM_KV_SPLIT_NO,                      "split.no"            },
    { LLM_KV_SPLIT_COUNT,                   "split.count"         },
    { LLM_KV_SPLIT_TENSORS_COUNT,           "split.tensors.count" },

    { LLM_KV_SSM_CONV_KERNEL,               "%s.ssm.conv_kernel"    },
    { LLM_KV_SSM_INNER_SIZE,                "%s.ssm.inner_size"     },
    { LLM_KV_SSM_STATE_SIZE,                "%s.ssm.state_size"     },
    { LLM_KV_SSM_TIME_STEP_RANK,            "%s.ssm.time_step_rank" },
    { LLM_KV_SSM_DT_B_C_RMS,                "%s.ssm.dt_b_c_rms" },

    { LLM_KV_WKV_HEAD_SIZE,                 "%s.wkv.head_size" },

    { LLM_KV_TOKENIZER_MODEL,                "tokenizer.ggml.model"                    },
    { LLM_KV_TOKENIZER_PRE,                  "tokenizer.ggml.pre"                      },
    { LLM_KV_TOKENIZER_LIST,                 "tokenizer.ggml.tokens"                   },
    { LLM_KV_TOKENIZER_TOKEN_TYPE,           "tokenizer.ggml.token_type"               },
    { LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,     "tokenizer.ggml.token_type_count"         },
    { LLM_KV_TOKENIZER_SCORES,               "tokenizer.ggml.scores"                   },
    { LLM_KV_TOKENIZER_MERGES,               "tokenizer.ggml.merges"                   },
    { LLM_KV_TOKENIZER_BOS_ID,               "tokenizer.ggml.bos_token_id"             },
    { LLM_KV_TOKENIZER_EOS_ID,               "tokenizer.ggml.eos_token_id"             },
    { LLM_KV_TOKENIZER_UNK_ID,               "tokenizer.ggml.unknown_token_id"         },
    { LLM_KV_TOKENIZER_SEP_ID,               "tokenizer.ggml.seperator_token_id"       },
    { LLM_KV_TOKENIZER_PAD_ID,               "tokenizer.ggml.padding_token_id"         },
    { LLM_KV_TOKENIZER_CLS_ID,               "tokenizer.ggml.cls_token_id"             },
    { LLM_KV_TOKENIZER_MASK_ID,              "tokenizer.ggml.mask_token_id"            },
    { LLM_KV_TOKENIZER_ADD_BOS,              "tokenizer.ggml.add_bos_token"            },
    { LLM_KV_TOKENIZER_ADD_EOS,              "tokenizer.ggml.add_eos_token"            },
    { LLM_KV_TOKENIZER_ADD_PREFIX,           "tokenizer.ggml.add_space_prefix"         },
    { LLM_KV_TOKENIZER_REMOVE_EXTRA_WS,      "tokenizer.ggml.remove_extra_whitespaces" },
    { LLM_KV_TOKENIZER_PRECOMPILED_CHARSMAP, "tokenizer.ggml.precompiled_charsmap"     },
    { LLM_KV_TOKENIZER_HF_JSON,              "tokenizer.huggingface.json"              },
    { LLM_KV_TOKENIZER_RWKV,                 "tokenizer.rwkv.world"                    },
    { LLM_KV_TOKENIZER_PREFIX_ID,            "tokenizer.ggml.prefix_token_id"          },
    { LLM_KV_TOKENIZER_SUFFIX_ID,            "tokenizer.ggml.suffix_token_id"          },
    { LLM_KV_TOKENIZER_MIDDLE_ID,            "tokenizer.ggml.middle_token_id"          },
    { LLM_KV_TOKENIZER_EOT_ID,               "tokenizer.ggml.eot_token_id"             },
    { LLM_KV_TOKENIZER_EOM_ID,               "tokenizer.ggml.eom_token_id"             },

    { LLM_KV_ADAPTER_TYPE,                  "adapter.type"       },
    { LLM_KV_ADAPTER_LORA_ALPHA,            "adapter.lora.alpha" },
};

struct LLM_KV {
    LLM_KV(llm_arch arch) : arch(arch) {}

    llm_arch arch;

    std::string operator()(llm_kv kv) const {
        return ::format(LLM_KV_NAMES.at(kv), LLM_ARCH_NAMES.at(arch));
    }
};

static llm_arch llm_arch_from_string(const std::string & name) {
    for (const auto & kv : LLM_ARCH_NAMES) { // NOLINT
        if (kv.second == name) {
            return kv.first;
        }
    }

    return LLM_ARCH_UNKNOWN;
}

struct naive_trie {
    naive_trie() : has_value(false), value(0) {
    }
    void insert(const char * key, size_t len, int32_t value = 0) {
        if (len == 0) {
            this->has_value = true;
            this->value = value;
            return;
        }
        char c = key[0];
        auto res = children.find(c);
        if (res != children.end()) {
            res->second.insert(key + 1, len - 1, value);
        } else {
            auto res = children.insert(std::make_pair(c, naive_trie()));
            res.first->second.insert(key + 1, len - 1, value);
        }
    }
    std::pair<const char *, size_t> get_longest_prefix(const char * key, size_t len, size_t offset = 0) {
        if (len == 0 || offset == len) {
            return std::make_pair(key, offset);
        }
        char c = key[offset];
        auto res = children.find(c);
        if (res != children.end()) {
            return res->second.get_longest_prefix(key, len, offset + 1);
        }

        return std::make_pair(key, offset);
    }
    const struct naive_trie * traverse(const char c) const {
        auto res = children.find(c);
        if (res != children.end()) {
            return &res->second;
        }

        return NULL;
    }
    std::map<char, struct naive_trie> children;
    bool has_value;
    llama_token value;
};

//
// impl
//

int llama_vocab::find_bpe_rank(const std::string & token_left, const std::string & token_right) const {
    assert(token_left.find(' ')   == std::string::npos);
    assert(token_left.find('\n')  == std::string::npos);
    assert(token_right.find(' ')  == std::string::npos);
    assert(token_right.find('\n') == std::string::npos);

    auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
    if (it == bpe_ranks.end()) {
        return -1;
    }

    return it->second;
}

static enum llama_vocab_type llama_vocab_get_type(const llama_vocab & vocab) {
    return vocab.type;
}

static bool llama_is_normal_token(const llama_vocab & vocab, llama_token id) {
    assert(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[id].attr & LLAMA_TOKEN_ATTR_NORMAL;
}

static bool llama_is_unknown_token(const llama_vocab & vocab, llama_token id) {
    assert(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[id].attr & LLAMA_TOKEN_ATTR_UNKNOWN;
}

static bool llama_is_control_token(const llama_vocab & vocab, llama_token id) {
    assert(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[id].attr & LLAMA_TOKEN_ATTR_CONTROL;
}

static bool llama_is_byte_token(const llama_vocab & vocab, llama_token id) {
    assert(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[id].attr & LLAMA_TOKEN_ATTR_BYTE;
}

static bool llama_is_user_defined_token(const llama_vocab & vocab, llama_token id) {
    assert(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[id].attr & LLAMA_TOKEN_ATTR_USER_DEFINED;
}

static bool llama_is_unused_token(const llama_vocab & vocab, llama_token id) {
    assert(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[id].attr & LLAMA_TOKEN_ATTR_UNUSED;
}

static uint8_t llama_token_to_byte(const llama_vocab & vocab, llama_token id) {
    assert(llama_vocab_get_type(vocab) != LLAMA_VOCAB_TYPE_NONE);
    assert(llama_is_byte_token(vocab, id));
    const auto & token_data = vocab.id_to_token.at(id);
    switch (llama_vocab_get_type(vocab)) {
        case LLAMA_VOCAB_TYPE_SPM:
        case LLAMA_VOCAB_TYPE_UGM: {
            auto buf = token_data.text.substr(3, 2);
            return strtol(buf.c_str(), NULL, 16);
        }
        case LLAMA_VOCAB_TYPE_BPE: {
            abort();
            //return unicode_utf8_to_byte(token_data.text); // TODO: why is this here after assert?
        }
        case LLAMA_VOCAB_TYPE_WPM: {
            abort();
        }
        default:
            abort();
    }
}

static void llama_escape_whitespace(std::string & text) {
    replace_all(text, " ", "\xe2\x96\x81");
}

static void llama_unescape_whitespace(std::string & word) {
    replace_all(word, "\xe2\x96\x81", " ");
}

struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

static_assert(std::is_trivially_copyable<llm_symbol>::value, "llm_symbol is not trivially copyable");

//
// SPM tokenizer
// original implementation:
// https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
//

struct llm_bigram_spm {
    struct comparator {
        bool operator()(llm_bigram_spm & l, llm_bigram_spm & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llm_bigram_spm>;
    using queue = std::priority_queue<llm_bigram_spm, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    float score;
    size_t size;
};

struct llm_tokenizer_spm {
    llm_tokenizer_spm(const llama_vocab & vocab) : vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llm_symbol sym;
            size_t len = unicode_len_utf8(text[offs]);
            sym.text = text.c_str() + offs;
            sym.n = std::min(len, text.size() - offs);
            offs += sym.n;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue.empty()) {
            auto bigram = work_queue.top();
            work_queue.pop();

            auto & left_sym = symbols[bigram.left];
            auto & right_sym = symbols[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //LLAMA_LOG_INFO("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols[i].next) {
            auto & symbol = symbols[i];
            resegment(symbol, output);
        }
    }

private:
    void resegment(llm_symbol & symbol, std::vector<llama_vocab::id> & output) {
        auto text = std::string(symbol.text, symbol.n);
        auto token = vocab.token_to_id.find(text);

        // Do we need to support is_unused?
        if (token != vocab.token_to_id.end()) {
            output.push_back((*token).second);
            return;
        }

        const auto p = rev_merge.find(text);

        if (p == rev_merge.end()) {
            // output any symbols that did not form tokens as bytes.
            output.reserve(output.size() + symbol.n);
            for (int j = 0; j < (int)symbol.n; ++j) {
                llama_vocab::id token_id = llama_byte_to_token_impl(vocab, symbol.text[j]);
                output.push_back(token_id);
            }
            return;
        }

        resegment(symbols[p->second.first],  output);
        resegment(symbols[p->second.second], output);
    }

    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols[left].text, symbols[left].n + symbols[right].n);
        auto token = vocab.token_to_id.find(text);

        if (token == vocab.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab.id_to_token.size()) {
            return;
        }

        const auto & tok_data = vocab.id_to_token[(*token).second];

        llm_bigram_spm bigram;
        bigram.left  = left;
        bigram.right = right;
        bigram.score = tok_data.score;
        bigram.size  = text.size();

        work_queue.push(bigram);

        // Do we need to support is_unused?
        rev_merge[text] = std::make_pair(left, right);
    }

    const llama_vocab & vocab;

    std::vector<llm_symbol> symbols;
    llm_bigram_spm::queue work_queue;

    std::map<std::string, std::pair<int, int>> rev_merge;
};

//
// BPE tokenizer
// adapted from https://github.com/cmp-nct/ggllm.cpp [MIT License]
// tried to simplify unicode stuff, so most likely does not work 100% correctly!
//

// TODO: there are a lot of common parts between spm and bpe tokenizers, should be refactored and reused

template<typename T, typename Container = std::vector<T>, typename Compare = std::less<typename Container::value_type>>
class llama_priority_queue : public std::priority_queue<T, Container, Compare> {
public:
    using std::priority_queue<T, Container, Compare>::priority_queue;

    T pop_move() {
        T item = std::move(this->c.front());
        std::pop_heap(this->c.begin(), this->c.end(), this->comp);
        this->c.pop_back();
        return item;
    }

    void pop() =  delete;
};

struct llm_bigram_bpe {
    struct comparator {
        bool operator()(const llm_bigram_bpe & l, const llm_bigram_bpe & r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = llama_priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    std::string text;
    int rank;
    size_t size;
};

struct llm_tokenizer_bpe {
    llm_tokenizer_bpe(const llama_vocab & vocab): vocab(vocab) {
        assert(vocab.type == LLAMA_VOCAB_TYPE_BPE);
        switch (vocab.type_pre) {
            case LLAMA_VOCAB_PRE_TYPE_LLAMA3:
                regex_exprs = {
                    // original regex from tokenizer.json
                    //"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",

                    // adapted: https://github.com/ggerganov/llama.cpp/pull/6920#issuecomment-2080233989
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_DBRX:
            case LLAMA_VOCAB_PRE_TYPE_SMAUG:
                regex_exprs = {
                    // same as llama3
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM:
                regex_exprs = {
                    "[\r\n]",
                    "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
                    "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
                    "\\s+$",
                    "[一-龥ࠀ-一가-퟿]+",
                    "\\p{N}+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER:
                regex_exprs = {
                    "[\r\n]",
                    "\\s?\\p{L}+",
                    "\\s?\\p{P}+",
                    "[一-龥ࠀ-一가-퟿]+",
                    "\\p{N}",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_FALCON:
                regex_exprs = {
                    "[\\p{P}\\$\\+<=>\\^~\\|`]+",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                    "[0-9][0-9][0-9]",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_STARCODER:
            case LLAMA_VOCAB_PRE_TYPE_REFACT:
            case LLAMA_VOCAB_PRE_TYPE_COMMAND_R:
            case LLAMA_VOCAB_PRE_TYPE_SMOLLM:
            case LLAMA_VOCAB_PRE_TYPE_CODESHELL:
            case LLAMA_VOCAB_PRE_TYPE_EXAONE:
                regex_exprs = {
                    "\\p{N}",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_GPT2:
            case LLAMA_VOCAB_PRE_TYPE_MPT:
            case LLAMA_VOCAB_PRE_TYPE_OLMO:
            case LLAMA_VOCAB_PRE_TYPE_JAIS:
                regex_exprs = {
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_STABLELM2:
            case LLAMA_VOCAB_PRE_TYPE_QWEN2:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_PORO:
            case LLAMA_VOCAB_PRE_TYPE_BLOOM:
            case LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH:
                regex_exprs = {
                    " ?[^(\\s|.,!?…。，、।۔،)]+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_CHATGLM4:
                regex_exprs = {
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_VIKING:
                regex_exprs = {
                    " ?[^(\\s|.,!?…。，、।۔،)]+",
                    "\\p{N}",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_TEKKEN:
                // original regex from tokenizer.json
                // "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                regex_exprs = {
                    "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            default:
                // default regex for BPE tokenization pre-processing
                regex_exprs = {
                    "[\\p{P}\\$\\+<=>\\^~\\|]+",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                    "\\p{N}+",
                    "[0-9][0-9][0-9]",
                };
                break;
        }
    }

    void append(const llama_vocab::id token_id, std::vector<llama_vocab::id> & output) const {
        output.push_back(token_id);
    }

    bool append_bos(std::vector<llama_vocab::id> & output) const {
        if (vocab.tokenizer_add_bos) {
            assert(vocab.special_bos_id != -1);
            output.push_back(vocab.special_bos_id);
            return true;
        }
        return false;
    }

    bool append_eos(std::vector<llama_vocab::id> & output) const {
        if (vocab.tokenizer_add_eos) {
            assert(vocab.special_eos_id != -1);
            output.push_back(vocab.special_eos_id);
            return true;
        }
        return false;
    }

    void check_double_bos_eos(const std::vector<llama_vocab::id> & output) const {
        if (vocab.tokenizer_add_bos && output.size() >= 2 && output[1] == vocab.special_bos_id) {
            fprintf(stderr,
                "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                "Are you sure this is what you want?\n", __FUNCTION__);
        }
        if (vocab.tokenizer_add_eos && output.size() >= 2 && *(output.end()-2) == vocab.special_eos_id) {
            fprintf(stderr,
                "%s: Added a EOS token to the prompt as specified by the model but the prompt "
                "also ends with a EOS token. So now the final prompt ends with 2 EOS tokens. "
                "Are you sure this is what you want?\n", __FUNCTION__);
        }
    }

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        int final_prev_index = -1;

        const auto word_collection = unicode_regex_split(text, regex_exprs);

        symbols_final.clear();

        for (auto & word : word_collection) {
            work_queue = llm_bigram_bpe::queue();
            symbols.clear();

            int index = 0;
            size_t offset = 0;

            if (vocab.tokenizer_ignore_merges && vocab.token_to_id.find(word) != vocab.token_to_id.end()) {
                symbols.emplace_back(llm_symbol{-1, -1, word.c_str(), word.size()});
                offset = word.size();
            }

            while (offset < word.size()) {
                llm_symbol sym;
                size_t char_len = std::min(word.size() - offset, (size_t) unicode_len_utf8(word[offset]));
                sym.text = word.c_str() + offset;
                sym.n = char_len;
                offset += sym.n;
                sym.prev = index - 1;
                sym.next = offset == word.size() ? -1 : index + 1;
                index++;
                symbols.emplace_back(sym);
            }
            for (size_t i = 1; i < symbols.size(); ++i) {
                add_new_bigram(i - 1, i);
            }

            // build token(s)
            while (!work_queue.empty()) {
                auto bigram = work_queue.pop_move();

                auto & left_symbol = symbols[bigram.left];
                auto & right_symbol = symbols[bigram.right];

                if (left_symbol.n == 0 || right_symbol.n == 0) {
                    continue;
                }
                std::string left_token = std::string(left_symbol.text, left_symbol.n);
                std::string right_token = std::string(right_symbol.text, right_symbol.n);
                if (left_token + right_token != bigram.text) {
                    continue;  // Skip this bigram if it's outdated
                }

                // merge the right sym into the left one
                left_symbol.n += right_symbol.n;
                right_symbol.n = 0;

                // remove the right sym from the chain
                left_symbol.next = right_symbol.next;
                if (right_symbol.next >= 0) {
                    symbols[right_symbol.next].prev = bigram.left;
                }

                add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
                add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
            }

            // add the finished tokens to the final list keeping correct order for next and prev
            for (auto & sym : symbols) {
                if (sym.n > 0) {
                    sym.prev = final_prev_index;
                    sym.next = -1;
                    if (final_prev_index != -1) {
                        symbols_final[final_prev_index].next = symbols_final.size();
                    }
                    symbols_final.emplace_back(sym);
                    final_prev_index = symbols_final.size() - 1;
                }
            }
        }

        symbols = symbols_final;

        if (!symbols.empty()) {
            for (int i = 0; i != -1; i = symbols[i].next) {
                auto & symbol = symbols[i];
                if (symbol.n == 0) {
                    continue;
                }

                const std::string str = std::string(symbol.text, symbol.n);
                const auto token = vocab.token_to_id.find(str);

                if (token == vocab.token_to_id.end()) {
                    for (auto j = str.begin(); j != str.end(); ++j) {
                        std::string byte_str(1, *j);
                        auto token_multibyte = vocab.token_to_id.find(byte_str);
                        if (token_multibyte != vocab.token_to_id.end()) {
                            output.push_back(token_multibyte->second);
                        }
                    }
                } else {
                    output.push_back((*token).second);
                }
            }
        }
    }

private:
    void add_new_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        std::string left_token  = std::string(symbols[left].text,  symbols[left].n);
        std::string right_token = std::string(symbols[right].text, symbols[right].n);

        int rank_found = -1;

        rank_found = vocab.find_bpe_rank(left_token, right_token);

        if (rank_found < 0) {
            return;
        }

        llm_bigram_bpe bigram;

        bigram.left  = left;
        bigram.right = right;
        bigram.text  = left_token + right_token;
        bigram.size  = left_token.size() + right_token.size();
        bigram.rank  = rank_found;

        work_queue.push(bigram);
    }

    const llama_vocab & vocab;

    std::vector<std::string> regex_exprs;

    std::vector<llm_symbol> symbols;
    std::vector<llm_symbol> symbols_final;

    llm_bigram_bpe::queue work_queue;
};

//
// WPM tokenizer
//

struct llm_tokenizer_wpm {
    llm_tokenizer_wpm(const llama_vocab & vocab): vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) const {
        const auto & token_map = vocab.token_to_id;

        // normalize and split by whitespace
        std::vector<std::string> words = preprocess(text);

        // bos token prepended already

        // find the longest tokens that form the words
        for (const std::string & word : words) {
            // skip empty words
            if (word.size() == 0) {
                continue;
            }

            // prepend phantom space
            const std::string word1 = "\xe2\x96\x81" + word;
            const int n = word1.size();

            const size_t current_tokens = output.size();

            // we're at the start of a new word
            // move through character position in word
            for (int i = 0; i < n; ++i) {
                // loop through possible match length
                bool match = false;
                for (int j = std::min(n, i + vocab.max_token_len + 1); j > i; j--) {
                    auto it = token_map.find(word1.substr(i, j - i));
                    if (it != token_map.end()) {
                        output.push_back(it->second);
                        match = true;
                        i = j - 1;
                        break;
                    }
                }

                if (!match) { // discard all
                    output.resize(current_tokens);
                    break;  // and discard next tokens
                }
            }

            // we didn't find any matches for this word
            if (current_tokens == output.size()) {
                output.push_back(vocab.special_unk_id);
            }
        }
    }

    // TODO: reduce string copies by using cpts_offs array
    std::vector<std::string> preprocess(const std::string & text) const {
        const std::vector<uint32_t> cpts_nfd = unicode_cpts_normalize_nfd(unicode_cpts_from_utf8(text));
        std::vector<std::string> words(1, "");

        for (const uint32_t cpt : cpts_nfd) {
            const auto flags = unicode_cpt_flags(cpt);

            if (flags.is_whitespace) {
                if (words.back().size()) {  // finish previous word if any
                    words.emplace_back();
                }
                continue;
            }

            assert (!flags.is_separator);
            if (cpt == 0 || cpt == 0xFFFD || flags.is_control) {
                continue;
            }

            const std::string s = unicode_cpt_to_utf8(unicode_tolower(cpt));
            if (flags.is_punctuation || ( cpt < 0x7F && flags.is_symbol ) || is_chinese_char(cpt)) {
                if (words.back().size()) {  // finish previous word if any
                    words.emplace_back();
                }
                words.back() = s;       // single char word
                words.emplace_back();   // start a new word
            } else {
                words.back() += s;  // append char to word
            }
        }

        if (!words.back().size()) {
            words.pop_back();
        }

        return words;
    }

    static bool is_chinese_char(uint32_t cpt) {
        return
            (cpt >= 0x04E00 && cpt <= 0x09FFF) ||
            (cpt >= 0x03400 && cpt <= 0x04DBF) ||
            (cpt >= 0x20000 && cpt <= 0x2A6DF) ||
            (cpt >= 0x2A700 && cpt <= 0x2B73F) ||
            (cpt >= 0x2B740 && cpt <= 0x2B81F) ||
            (cpt >= 0x2B920 && cpt <= 0x2CEAF) || // this should be 0x2B820 but in hf rust code it is 0x2B920
            (cpt >= 0x0F900 && cpt <= 0x0FAFF) ||
            (cpt >= 0x2F800 && cpt <= 0x2FA1F);
            //(cpt >= 0x3000  && cpt <= 0x303F)  ||
            //(cpt >= 0xFF00  && cpt <= 0xFFEF);
    }

    const llama_vocab & vocab;
};

//
// UGM tokenizer
//

struct llm_tokenizer_ugm {
    llm_tokenizer_ugm(const llama_vocab & vocab) : vocab(vocab) {
        if (vocab.precompiled_charsmap.size() > 0) {
            size_t charsmap_offset = 0;

            // First four bytes of precompiled_charsmap contains length of binary
            // blob containing XOR-compressed compact double array (XCDA) entries
            uint32_t xcda_blob_size = *(const uint32_t *) &vocab.precompiled_charsmap[0];
            charsmap_offset += sizeof(xcda_blob_size);
            if (xcda_blob_size + charsmap_offset >= vocab.precompiled_charsmap.size()) {
                throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
            }

            // Next xcda_blob_size bytes contain entries of XOR-compressed compact
            // double array (XCDA). Each entry is bit-packed into a 32-bit integer.
            xcda_array = (const uint32_t *) &vocab.precompiled_charsmap[charsmap_offset];
            xcda_array_size = xcda_blob_size / sizeof(uint32_t);
            charsmap_offset += xcda_blob_size;

            // Remaining bytes of precompiled charsmap contain null-terminated
            // replacement strings for prefixes matched by the XCDA.
            prefix_replacements = &vocab.precompiled_charsmap[charsmap_offset];
            prefix_replacements_size = vocab.precompiled_charsmap.size() - charsmap_offset;
        }

        for (unsigned int id = 0; id < vocab.id_to_token.size(); ++id) {
            const auto &token_data = vocab.id_to_token[id];

            if (llama_is_normal_token(vocab, id)) {
                min_score = std::min<float>(min_score, token_data.score);
                max_score = std::max<float>(max_score, token_data.score);
            }

            if (llama_is_normal_token(vocab, id) ||
                llama_is_user_defined_token(vocab, id) ||
                llama_is_unused_token(vocab, id)) {
                token_matcher.insert(token_data.text.data(), token_data.text.size(), id);
            }

            if (llama_is_user_defined_token(vocab, id)) {
                user_defined_token_matcher.insert(token_data.text.data(), token_data.text.size());
            }
        }

        unknown_token_score = min_score - unknown_token_score_penalty;
    }

    /* This implementation is based on SentencePiece optimized Viterbi algorithm for
     * unigram language models. The general idea is to:
     * - move along the input sequence in steps of one UTF code point,
     * - at each step find all possible tokenizations of the prefix by
     *   traversing the tokens trie,
     * - for each tokenization store the best one so far (by higher score)
     * - use the position in sequence after given token as an index to store
     *   results
     * - if there was no valid tokenization of the current UTF code point
     *   then use unknown token with additional score penalty
     * After processing the whole sequence we backtrack from the end to get
     * the best tokenization.
    */
    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        // get current size of output (for reversal later)
        size_t output_size = output.size();

        // normalize the input first
        std::string normalized;
        normalize(text, &normalized);
        size_t input_len = normalized.size();
        if (input_len == 0) {
            return;
        }

        // initialize score_sum to -FLT_MAX so it will be always lower than sums of token scores
        std::vector<struct best_tokenization> tokenization_results(input_len + 1, {vocab.special_unk_id, 0, -FLT_MAX});
        // at the beginning tokenization score is zero
        tokenization_results[0] = { vocab.special_unk_id, 0, 0 };

        for (size_t input_offset = 0; input_offset < input_len;) {
            size_t prefix_offset = input_offset;
            // calculate how many code units are in the currently processed UTF code point
            size_t n_utf8_code_units = std::min<size_t>(unicode_len_utf8(normalized[input_offset]), input_len - input_offset);

            // traverse the token matcher trie to find a matching token
            bool single_codepoint_token_found = false;
            const struct best_tokenization & current_best = tokenization_results[input_offset];
            const struct naive_trie * node = token_matcher.traverse(normalized[prefix_offset++]);

            while (prefix_offset <= input_len && node != NULL) {
                // check if we found valid token in prefix
                if (node->has_value) {
                    // check if it corresponds to the whole UTF code point
                    if (prefix_offset - input_offset == n_utf8_code_units) {
                        single_codepoint_token_found = true;
                    }
                    llama_token token_id = node->value;
                    const auto & token_data = vocab.id_to_token[token_id];

                    // we set the user-defined token scores to 0 to make them more likely to be selected
                    // (normal token scores are log probabilities, so they are negative)
                    // score type is double here to make tokenization results exactly
                    // the same as in the HF tokenizer using SentencePiece
                    const double token_score = llama_is_user_defined_token(vocab, token_id) ? 0.0 : token_data.score;
                    const double challenger_score = current_best.score_sum + token_score;
                    struct best_tokenization & current_champ = tokenization_results[prefix_offset];
                    if (challenger_score > current_champ.score_sum) {
                        struct best_tokenization challenger = { token_id, input_offset, (float) challenger_score };
                        current_champ = challenger;
                    }
                }
                node = node->traverse(normalized[prefix_offset++]);
            }

            // if we didn't find a valid token corresponding to the whole UTF code point
            // then use unknown token as the tokenization of this UTF code point
            if (!single_codepoint_token_found) {
                const double challenger_score = current_best.score_sum + unknown_token_score;
                prefix_offset = input_offset + n_utf8_code_units;
                struct best_tokenization & current_champ = tokenization_results[prefix_offset];
                if (challenger_score > current_champ.score_sum) {
                    struct best_tokenization challenger = { vocab.special_unk_id, input_offset, (float) challenger_score };
                    current_champ = challenger;
                }
            }

            // move to the next UTF code point
            input_offset += n_utf8_code_units;
        }

        // now backtrack from the end to gather token ids of the best tokenization
        // merge sequences of consecutive unknown tokens into single unknown tokens
        bool is_prev_unknown = false;
        for (struct best_tokenization & tokenization = tokenization_results[input_len]; ; tokenization = tokenization_results[tokenization.input_offset]) {
            bool is_unknown = tokenization.token_id == vocab.special_unk_id;
            if (!(is_prev_unknown && is_unknown)) {
                output.push_back(tokenization.token_id);
            }
            if (tokenization.input_offset == 0) {
                break;
            }
            is_prev_unknown = is_unknown;
        }

        // reverse the output since we added tokens starting from the end of the input
        std::reverse(output.begin() + output_size, output.end());
    }

private:
    const llama_vocab & vocab;

    // helper structure for returning normalization results
    struct normalization_result {
        const char * normalized;
        size_t normalized_len;
        size_t consumed_input;
    };

    void normalize(const std::string& input, std::string * normalized) {
        normalized->clear();
        normalized->reserve(input.size() * 3);

        const std::string space = vocab.tokenizer_escape_whitespaces ? escaped_space : " ";

        bool shall_prepend_space = !vocab.tokenizer_treat_whitespace_as_suffix && vocab.tokenizer_add_space_prefix;
        bool shall_append_space = vocab.tokenizer_treat_whitespace_as_suffix && vocab.tokenizer_add_space_prefix;
        bool shall_merge_spaces = vocab.tokenizer_remove_extra_whitespaces;

        bool is_space_prepended = false;
        bool processing_non_ws = false;

        size_t input_len = input.size();

        for (size_t input_offset = 0; input_offset < input_len; ) {
            auto norm_res = normalize_prefix(input, input_offset);
            for (size_t i = 0; i < norm_res.normalized_len; i++) {
                char c = norm_res.normalized[i];
                if (c != ' ') {
                    if (!processing_non_ws) {
                        processing_non_ws = true;
                        if ((shall_prepend_space && !is_space_prepended) || shall_merge_spaces) {
                            normalized->append(space);
                            is_space_prepended = true;
                        }
                    }
                    normalized->push_back(c);
                } else {
                    if (processing_non_ws) {
                        processing_non_ws = false;
                    }
                    if (!shall_merge_spaces) {
                        normalized->append(space);
                    }
                }
            }

            input_offset += norm_res.consumed_input;
        }

        if (shall_append_space) {
            normalized->append(space);
        }
    }

    /*
     * This structure is a view wrapper for XOR-compressed double array (XCDA)
     * See Shunsuke Kanda (2018). Space- and Time-Efficient String Dictionaries.
     * Each bit-packed entry contains:
     * - BASE array value in bits 10-30
     * - LCHECK array value in bits 0-7
     * - LEAF array value in bit 9
     * Entries containing indexes of replacement sequences have set bit 31
     */
    struct xcda_array_view {
    public:
        xcda_array_view(const uint32_t * xcda_array, size_t xcda_array_size) : xcda_array(xcda_array), xcda_array_size(xcda_array_size) {
        }
        uint32_t get_base(size_t index) {
            uint32_t packed_node = get_node(index);
            return (packed_node >> 10) << ((packed_node & (1U << 9)) >> 6);
        }
        uint32_t get_lcheck(size_t index) {
            uint32_t packed_node = get_node(index);
            return packed_node & ((1U << 31) | 0xff);
        }
        bool get_leaf(size_t index) {
            uint32_t packed_node = get_node(index);
            return (packed_node >> 8) & 1;
        }
        uint32_t get_value(size_t index) {
            uint32_t packed_node = get_node(index);
            return packed_node & ((1U << 31) - 1);
        }
    private:
        uint32_t get_node(size_t index) {
            if (index > xcda_array_size) {
                throw std::runtime_error("Index out of array bounds in XCDA array!");
            }
            return xcda_array[index];
        }
        const uint32_t * xcda_array;
        size_t xcda_array_size;
    };

    struct normalization_result normalize_prefix(const std::string & input, size_t input_offset) {
        if (input_offset == input.size()) {
            return { &input[input_offset], 0, 0 };
        }

        // if input prefix matches some user-defined token return this token as normalization result
        auto user_defined_token_match = user_defined_token_matcher.get_longest_prefix(&input[input_offset], input.size() - input_offset);
        if (user_defined_token_match.second > 0) {
            return { &input[input_offset], user_defined_token_match.second, user_defined_token_match.second };
        }

        size_t longest_prefix_length = 0;
        size_t longest_prefix_offset = 0;

        if (xcda_array_size > 0) {
            struct xcda_array_view xcda_view(xcda_array, xcda_array_size);

            // Find the longest normalized sequence matching the input prefix by walking
            // the XOR-compressed compact double array (XCDA) starting from the root node
            // We find the index of the next node by calculating BASE[s] ^ c where s is
            // the index of the previous node and c is a numerical character value
            uint32_t node_index = 0;
            // get BASE of the root node
            node_index = xcda_view.get_base(node_index);
            for (size_t prefix_offset = input_offset; prefix_offset < input.size(); prefix_offset++) {
                unsigned char c = input[prefix_offset];
                if (c == 0) {
                    break;
                }
                node_index ^= c;
                // if value of LCHECK is not c it means that this is not a child of
                // the previous node, so we stop matching
                if (xcda_view.get_lcheck(node_index) != c) {
                    break;
                }
                bool is_leaf = xcda_view.get_leaf(node_index);
                // get BASE of the current node
                node_index ^= xcda_view.get_base(node_index);
                // if LEAF of the current node is true, it means that its BASE points to the node
                // containing index of replacement sequence for currently matched input prefix
                if (is_leaf)
                {
                    longest_prefix_length = prefix_offset - input_offset + 1;
                    // get index of replacement sequence for currently matched input prefix
                    longest_prefix_offset = xcda_view.get_value(node_index);
                }
            }
        }

        if (longest_prefix_length > 0) {
            // we have a match, so return the replacement sequence
            if (longest_prefix_offset >= prefix_replacements_size) {
                throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
            }
            const char * prefix_replacement = &prefix_replacements[longest_prefix_offset];
            return { prefix_replacement, strlen(prefix_replacement), longest_prefix_length };
        } else {
            // check if the input prefix contains a valid sequence of UTF-8 code units
            try {
                // if yes, return this sequence unmodified
                size_t prefix_offset = input_offset;
                unicode_cpt_from_utf8(input, prefix_offset);
                return { &input[input_offset], prefix_offset - input_offset, prefix_offset - input_offset };
            } catch (std::invalid_argument & /*ex*/) {
                // if no, consume 1 byte and return U+FFFD - REPLACEMENT CHARACTER
                return { "\xEF\xBF\xBD", 3, 1 };
            }
        }
    }

    // escaped space symbol - U+2581 (Lower One Eighth Block)
    const std::string escaped_space = "\xE2\x96\x81";

    const char * prefix_replacements = NULL;
    size_t prefix_replacements_size = 0;

    const uint32_t * xcda_array = NULL;
    size_t xcda_array_size = 0;

    struct naive_trie user_defined_token_matcher;

    // this structure stores the best tokenization so far at input_offset
    struct best_tokenization {
        llama_token token_id;
        size_t input_offset;
        float score_sum;
    };

    float min_score = FLT_MAX;
    float max_score = -FLT_MAX;

    float unknown_token_score_penalty = 10.0;
    float unknown_token_score;

    struct naive_trie token_matcher;
};

//
// RWKV tokenizer
//

static std::vector<uint8_t> llama_unescape_rwkv_token(const std::string & escaped) {
    std::vector<uint8_t> output;
    output.reserve(escaped.size());

    // Parser state
    bool escaping = false;
    uint8_t hex_remaining = 0;
    uint8_t hex_acc = 0;

    // Step through characters, performing parsing
    for (const char & c : escaped) {
        // If we're parsing a hex code, interpret the next character
        if (hex_remaining != 0) {
            uint8_t value = (c >= 'a') ? (c - 'a' + 10) : (c - '0');
            hex_acc = (hex_acc << 4) + value;

            hex_remaining -= 1;
            if (hex_remaining == 0) {
                output.push_back(hex_acc);
                hex_acc = 0;
            }

            continue;
        }

        // If we got an escape character, interpret it
        if (escaping) {
            if (c == 't') {
                output.push_back('\t');
            } else if (c == 'n') {
                output.push_back('\n');
            } else if (c == 'r') {
                output.push_back('\r');
            } else if (c == 'x') {
                hex_remaining = 2;
            } else {
                output.push_back(c);
            }

            escaping = false;
            continue;
        }

        if (c == '\\') {
            escaping = true;
            continue;
        }

        output.push_back(c);
    }

    return output;
}

struct llm_tokenizer_rwkv {
    llm_tokenizer_rwkv(const llama_vocab & vocab): vocab(vocab) {
        // RWKV supports arbitrary byte tokens, but the vocab struct only supports string tokens.
        // For now, we decode the vocab here into the lookup we'll use for tokenization.

        // build trie
        for (unsigned int id = 0; id < vocab.id_to_token.size(); ++id) {
            const auto & token = vocab.id_to_token[id];
            const auto data = llama_unescape_rwkv_token(token.text);
            token_matcher.insert((const char *) data.data(), data.size(), id);
        }
    }

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        uint32_t position = 0;

        while (position < text.size()) {
            const struct naive_trie * node = token_matcher.traverse(text[position]);
            if (node == NULL) {
                // no matching token found, add unknown token
                output.push_back(vocab.special_unk_id);
                position += 1;
                continue;
            }

            // traverse the trie to find the longest matching token
            uint32_t token_id = 0;
            uint32_t token_length = 0;
            while (node != NULL) {
                if (node->has_value) {
                    token_id = node->value;
                    token_length = position + 1;
                }
                node = node->traverse(text[++position]);
            }

            // add the longest matching token
            output.push_back(token_id);
            position = token_length;
        }
    }

    const llama_vocab & vocab;

    struct naive_trie token_matcher;
};

//
// (de-) tokenize
//

typedef enum FRAGMENT_BUFFER_VARIANT_TYPE {
    FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN,
    FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT
} FRAGMENT_BUFFER_VARIANT_TYPE;

struct fragment_buffer_variant {
    fragment_buffer_variant(llama_vocab::id _token)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN),
        token(_token),
        raw_text(_dummy),
        offset(0),
        length(0) {}

    fragment_buffer_variant(const std::string & _raw_text, int64_t _offset, int64_t _length)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT),
        token((llama_vocab::id) - 1),
        raw_text(_raw_text),
        offset(_offset),
        length(_length){
            assert(_offset >= 0);
            assert(_length >= 1);
            assert(offset + length <= raw_text.length());
        }

    const FRAGMENT_BUFFER_VARIANT_TYPE type;
    const llama_vocab::id token;
    const std::string _dummy;
    const std::string & raw_text;
    const uint64_t offset;
    const uint64_t length;
};

// #define PRETOKENIZERDEBUG

static void tokenizer_st_partition(const llama_vocab & vocab, std::forward_list<fragment_buffer_variant> & buffer, bool parse_special) {
    // for each special token
    for (const llama_vocab::id special_id : vocab.cache_special_tokens) {
        const auto & data = vocab.id_to_token[special_id];
        const auto & special_token = data.text;

        if (!parse_special && (data.attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_UNKNOWN))) {
            // Ignore control and unknown tokens when parse_special == false
            continue;
            // User-defined tokens are still pre-tokenized before everything else
            // ref: https://github.com/huggingface/tokenizers/blob/fdd26ba9a3f0c133427aab0423888cbde91362d7/tokenizers/src/tokenizer/mod.rs#L726
            // This is mostly relevant for neox-style tokenizers (mpt, olmo, stablelm, etc.)
        }

        // for each text fragment
        std::forward_list<fragment_buffer_variant>::iterator it = buffer.begin();
        while (it != buffer.end()) {
            auto & fragment = (*it);

            // if a fragment is text ( not yet processed )
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                auto & raw_text = fragment.raw_text;

                auto raw_text_base_offset = fragment.offset;
                auto raw_text_base_length = fragment.length;

                // loop over the text
                while (true) {
                    // find the first occurrence of a given special token in this fragment
                    //  passing offset argument only limit the "search area" but match coordinates
                    //  are still relative to the source full raw_text
                    auto match = raw_text.find(special_token, raw_text_base_offset);

                    // no occurrences found, stop processing this fragment for a given special token
                    if (match == std::string::npos) break;

                    // check if match is within bounds of offset <-> length
                    if (match + special_token.length() > raw_text_base_offset + raw_text_base_length) break;

#ifdef PRETOKENIZERDEBUG
                    LLAMA_LOG_WARN("FF: (%ld %ld %ld) '%s'\n", raw_text->length(), raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    auto source = std::distance(buffer.begin(), it);

                    // if match is further than base offset
                    //  then we have some text to the left of it
                    if (match > raw_text_base_offset) {
                        // left
                        const int64_t left_reminder_offset = raw_text_base_offset + 0;
                        int64_t left_reminder_length = match - raw_text_base_offset;

                        if (data.attr & LLAMA_TOKEN_ATTR_LSTRIP) {
                            while (left_reminder_length > 0 && isspace(raw_text[left_reminder_offset + left_reminder_length - 1])) {
                                left_reminder_length--;
                            }
                        }

                        if (left_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, left_reminder_offset, left_reminder_length);
                            it++;
                        }

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("FL: (%ld %ld) '%s'\n", left_reminder_offset, left_reminder_length, raw_text->substr(left_reminder_offset, left_reminder_length).c_str());
#endif
                    }

                    // special token
                    buffer.emplace_after(it, special_id);
                    it++;

                    // right
                    if (match + special_token.length() < raw_text_base_offset + raw_text_base_length) {
                        int64_t right_reminder_offset = match + special_token.length();
                        int64_t right_reminder_length = raw_text_base_length - ((match - raw_text_base_offset) + special_token.length());

                        if (data.attr & LLAMA_TOKEN_ATTR_RSTRIP) {
                            while (right_reminder_length > 0 && isspace(raw_text[right_reminder_offset])) {
                                right_reminder_offset++;
                                right_reminder_length--;
                            }
                        }

                        if (right_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, right_reminder_offset, right_reminder_length);
                            it++;
                        }

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("FR: (%ld %ld) '%s'\n", right_reminder_offset, right_reminder_length, raw_text->substr(right_reminder_offset, right_reminder_length).c_str());
#endif

                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source-1)));
                        }

                        // repeat for the right side
                        raw_text_base_offset = right_reminder_offset;
                        raw_text_base_length = right_reminder_length;

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("RR: (%ld %ld) '%s'\n", raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    } else {
                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source-1)));
                        }
                        break;
                    }
                }
            }
            it++;
        }
    }
}

std::vector<llama_vocab::id> llama_tokenize_internal(const llama_vocab & vocab, std::string raw_text, bool add_special, bool parse_special) {
    std::vector<llama_vocab::id> output;
    std::forward_list<fragment_buffer_variant> fragment_buffer;

    if (!raw_text.empty()) {
        fragment_buffer.emplace_front(raw_text, 0, raw_text.length());
        tokenizer_st_partition(vocab, fragment_buffer, parse_special);
    }

    switch (vocab.type) {
        case LLAMA_VOCAB_TYPE_SPM:
            {
                // OG tokenizer behavior:
                //
                // tokenizer.encode('', add_special_tokens=True)  returns [1]
                // tokenizer.encode('', add_special_tokens=False) returns []

                bool is_prev_special = true;  // prefix with space if first token

                if (add_special && vocab.tokenizer_add_bos) {
                    assert(vocab.special_bos_id != -1);
                    output.push_back(vocab.special_bos_id);
                    is_prev_special = true;
                }

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);

                        // prefix with space if previous is special
                        if (vocab.tokenizer_add_space_prefix && is_prev_special) {
                            raw_text = " " + raw_text;
                        }

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                        llm_tokenizer_spm tokenizer(vocab);
                        llama_escape_whitespace(raw_text);
                        tokenizer.tokenize(raw_text, output);
                        is_prev_special = false;
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                        is_prev_special = true;
                    }
                }

                if (add_special && vocab.tokenizer_add_bos && output.size() >= 2 && output[1] == vocab.special_bos_id) {
                    fprintf(stderr,
                        "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                        "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                        "Are you sure this is what you want?\n", __FUNCTION__);
                }

                if (add_special && vocab.tokenizer_add_eos) {
                    assert(vocab.special_eos_id != -1);
                    output.push_back(vocab.special_eos_id);
                }
            } break;
        case LLAMA_VOCAB_TYPE_BPE:
            {
                llm_tokenizer_bpe tokenizer(vocab);

                if (add_special) {
                    tokenizer.append_bos(output);
                }
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                        tokenizer.tokenize(raw_text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        tokenizer.append(fragment.token, output);
                    }
                }

                if (add_special) {
                    tokenizer.append_eos(output);
                    tokenizer.check_double_bos_eos(output);
                }
            } break;
        case LLAMA_VOCAB_TYPE_WPM:
            {
                if (add_special) {
                    assert(vocab.special_cls_id != -1);
                    output.push_back(vocab.special_cls_id);
                }

                llm_tokenizer_wpm tokenizer(vocab);

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                        tokenizer.tokenize(raw_text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }

                if (add_special) {
                    assert(vocab.special_sep_id != -1);
                    output.push_back(vocab.special_sep_id);
                }
            } break;
        case LLAMA_VOCAB_TYPE_UGM:
            {
                llm_tokenizer_ugm tokenizer(vocab);

                if (add_special && vocab.tokenizer_add_bos != 0) {
                    assert(vocab.special_bos_id != -1);
                    output.push_back(vocab.special_bos_id);
                }

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);
#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                        tokenizer.tokenize(raw_text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }

                if (add_special && vocab.tokenizer_add_bos != 0 && output.size() >= 2 && output[1] == vocab.special_bos_id) {
                    fprintf(stderr,
                        "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                        "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                        "Are you sure this is what you want?\n", __FUNCTION__);
                }

                if (add_special && vocab.tokenizer_add_eos == 1) {
                    assert(vocab.special_eos_id != -1);
                    output.push_back(vocab.special_eos_id);
                }
            } break;
        case LLAMA_VOCAB_TYPE_RWKV:
            {
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif

                        llm_tokenizer_rwkv tokenizer(vocab);
                        tokenizer.tokenize(raw_text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }
            } break;
        case LLAMA_VOCAB_TYPE_NONE:
            abort();
    }

    return output;
}

llama_token llama_byte_to_token_impl(const llama_vocab & vocab, uint8_t ch) {
    assert(llama_vocab_get_type(vocab) != LLAMA_VOCAB_TYPE_NONE);
    static const char * hex = "0123456789ABCDEF";
    switch (llama_vocab_get_type(vocab)) {
        case LLAMA_VOCAB_TYPE_SPM:
        case LLAMA_VOCAB_TYPE_UGM: {
            const char buf[7] = { '<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>', 0 };
            auto token = vocab.token_to_id.find(buf);
            if (token != vocab.token_to_id.end()) {
                return (*token).second;
            }
            // Try to fall back to just the byte as a string
            const char buf2[2] = { (char)ch, 0 };
            return vocab.token_to_id.at(buf2);
        }
        case LLAMA_VOCAB_TYPE_WPM:
        case LLAMA_VOCAB_TYPE_BPE: {
            return vocab.token_to_id.at(unicode_byte_to_utf8(ch));
        }
        default:
            abort();
    }
}

const char * llama_token_get_text_impl(const struct llama_vocab & vocab, llama_token token) {
    assert(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[token].text.c_str();
}

float llama_token_get_score_impl(const struct llama_vocab & vocab, llama_token token) {
    assert(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[token].score;
}

llama_token_attr llama_token_get_attr_impl(const struct llama_vocab & vocab, llama_token token) {
    assert(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[token].attr;
}

bool llama_token_is_eog_impl(const struct llama_vocab & vocab, llama_token token) {
    return token != -1 && (
        token == llama_token_eos_impl(vocab) ||
        token == llama_token_eot_impl(vocab) ||
        token == llama_token_eom_impl(vocab)
    );
}

bool llama_token_is_control_impl(const struct llama_vocab & vocab, llama_token token) {
    return llama_is_control_token(vocab, token);
}

llama_token llama_token_bos_impl(const struct llama_vocab & vocab) {
    return vocab.special_bos_id;
}

llama_token llama_token_eos_impl(const struct llama_vocab & vocab) {
    return vocab.special_eos_id;
}

llama_token llama_token_cls_impl(const struct llama_vocab & vocab) {
    return vocab.special_cls_id;
}

llama_token llama_token_sep_impl(const struct llama_vocab & vocab) {
    return vocab.special_sep_id;
}

llama_token llama_token_nl_impl(const struct llama_vocab & vocab) {
    return vocab.linefeed_id;
}

llama_token llama_token_pad_impl(const struct llama_vocab & vocab) {
    return vocab.special_pad_id;
}

bool llama_add_bos_token_impl(const struct llama_vocab & vocab) {
    return vocab.tokenizer_add_bos;
}

bool llama_add_eos_token_impl(const struct llama_vocab & vocab) {
    return vocab.tokenizer_add_eos;
}

llama_token llama_token_prefix_impl(const struct llama_vocab & vocab) {
    return vocab.special_prefix_id;
}

llama_token llama_token_middle_impl(const struct llama_vocab & vocab) {
    return vocab.special_middle_id;
}

llama_token llama_token_suffix_impl(const struct llama_vocab & vocab) {
    return vocab.special_suffix_id;
}

llama_token llama_token_eot_impl(const struct llama_vocab & vocab) {
    return vocab.special_eot_id;
}

llama_token llama_token_eom_impl(const struct llama_vocab & vocab) {
    return vocab.special_eom_id;
}

int32_t llama_tokenize_impl(
    const struct llama_vocab & vocab,
                  const char * text,
                     int32_t   text_len,
                 llama_token * tokens,
                     int32_t   n_tokens_max,
                        bool   add_special,
                        bool   parse_special) {
    auto res = llama_tokenize_internal(vocab, std::string(text, text_len), add_special, parse_special);
    if (n_tokens_max < (int) res.size()) {
        // LLAMA_LOG_ERROR("%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

static std::string llama_decode_text(const std::string & text) {
    std::string decoded_text;

    const auto cpts = unicode_cpts_from_utf8(text);
    for (const auto cpt : cpts) {
        const auto utf8 = unicode_cpt_to_utf8(cpt);
        try {
            decoded_text += unicode_utf8_to_byte(utf8);
        } catch (const std::out_of_range & /*e*/) {
            decoded_text += "[UNK_BYTE_0x";
            for (const auto c : utf8) {
                decoded_text += format("%02x", (uint8_t) c);
            }
            decoded_text += text + "]";
        }
    }

    return decoded_text;
}

// does not write null-terminator to buf
int32_t llama_token_to_piece_impl(const struct llama_vocab & vocab, llama_token token, char * buf, int32_t length, int32_t lstrip, bool special) {
    // ref: https://github.com/ggerganov/llama.cpp/pull/7587#discussion_r1620983843
    static const int attr_special = LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_CONTROL;
    const llama_token_attr attr = llama_token_get_attr_impl(vocab, token);
    if (!special && (attr & attr_special)) {
        return 0;
    }

    // copy piece chars to output text buffer
    // skip up to 'lstrip' leading spaces before copying
    auto _try_copy = [=] (const char * token, size_t size) -> int32_t {
        for (int32_t i = 0; i < lstrip && size && *token == ' '; ++i) {
            token++;
            size--;
        }
        if (length < (int32_t)size) {
            return -(int32_t) size;
        }
        memcpy(buf, token, size);
        return (int32_t) size;
    };

    // if we have a cache - use it
    {
        const auto & cache = vocab.cache_token_to_piece;

        if (!cache.empty()) {
            const auto & result = cache.at(token);
            return _try_copy(result.data(), result.size());
        }
    }

    if (0 <= token && token < (int32_t) vocab.id_to_token.size()) {
        const std::string & token_text = vocab.id_to_token[token].text;
        switch (llama_vocab_get_type(vocab)) {
            case LLAMA_VOCAB_TYPE_WPM:
            case LLAMA_VOCAB_TYPE_SPM:
            case LLAMA_VOCAB_TYPE_UGM: {
                // NOTE: we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                if (attr & (attr_special | LLAMA_TOKEN_ATTR_USER_DEFINED)) {
                    return _try_copy(token_text.data(), token_text.size());
                } else if (attr & LLAMA_TOKEN_ATTR_NORMAL) {
                    std::string result = token_text;
                    llama_unescape_whitespace(result);
                    return _try_copy(result.data(), result.size());
                } else if (attr & LLAMA_TOKEN_ATTR_BYTE) {
                    char byte = (char) llama_token_to_byte(vocab, token);
                    return _try_copy((char*) &byte, 1);
                }
                break;
            }
            case LLAMA_VOCAB_TYPE_BPE: {
                // NOTE: we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                if (attr & (attr_special | LLAMA_TOKEN_ATTR_USER_DEFINED)) {
                    return _try_copy(token_text.data(), token_text.size());
                } else if (attr & LLAMA_TOKEN_ATTR_NORMAL) {
                    std::string result = llama_decode_text(token_text);
                    return _try_copy(result.data(), result.size());
                }
                break;
            }
            case LLAMA_VOCAB_TYPE_RWKV: {
                std::vector<uint8_t> result = llama_unescape_rwkv_token(token_text);

                // If we don't have enough space, return an error
                if (result.size() > (size_t)length) {
                    return -(int)result.size();
                }

                memcpy(buf, result.data(), result.size());
                return (int)result.size();
            }
            default:
                abort();
        }
    }

    return 0;
}

int32_t llama_detokenize_impl(
        const struct llama_vocab & vocab,
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special) {
    int32_t avail = text_len_max;
    int32_t total = 0;

    // remove the leading space
    bool remove_space = vocab.tokenizer_add_space_prefix;

    if (remove_special && vocab.tokenizer_add_bos) {
        if (n_tokens > 0 && tokens[0] == vocab.special_bos_id) {
            remove_space = false;
            n_tokens--;
            tokens++;
        }
    }

    if (remove_special && vocab.tokenizer_add_eos) {
        if (n_tokens > 0 && tokens[n_tokens-1] == vocab.special_eos_id) {
            n_tokens--;
        }
    }

    for (int32_t i = 0; i < n_tokens; ++i) {
        assert(avail >= 0);
        int32_t n_chars = llama_token_to_piece_impl(vocab, tokens[i], text, avail, remove_space, unparse_special);
        remove_space = false;
        if (n_chars < 0) {
            avail = 0;
            total -= n_chars;
        } else if (n_chars > 0) {
            avail -= n_chars;
            text  += n_chars;
            total += n_chars;
        }
    }

    if (total > text_len_max) {
        return -total;
    }

    if (vocab.tokenizer_clean_spaces) {
        text -= total;  // restart text

        // first pass: characters ?!.,  //TODO: where do these characters come from?
        const int32_t total1 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total1; ++i) {
            const char x = text[i];
            if (text[i - 1] == ' ') {
                if (x == '?' || x == '!' || x == '.' || x == ',') {  // " ?", " !", " .", " ,"
                    total--;  // remove space
                }
            }
            text[total++] = x;
        }

        // second pass: strip single apostrophe between spaces
        const int32_t total2 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total2; ++i) {
            const char x = text[i];
            if (x == '\'' && i + 1 < total2 && text[i - 1] == ' ' && text[i + 1] == ' ') {  // " ' "
                total--;           // remove prev space
                text[++i] = '\0';  // remove next space
            }
            text[total++] = x;
        }

        // third pass: apostrophe contractions  //NOTE: this makes sense?
        const int32_t total3 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total3; ++i) {
            const char x = text[i];
            if (text[i - 1] == ' ') {
                if (x == '\'' && i + 1 < total3) {
                    const char x1 = text[i + 1];
                    if (x1 == 't' || x1 == 'd') {  // " 't", " 'd"
                        //total--;  // remove space
                    } else if (x1 == 's' || x1 == 'm') {  // " 's", " 'm"
                        total--;  // remove space
                    } else if (i + 2 < total3) {
                        const char x2 = text[i + 2];
                        if ((x1 == 'l' && x2 == 'l')) {  // " 'll"
                            //total--;  // remove space
                        } else if ((x1 == 'r' && x2 == 'e') || (x1 == 'v' && x2 == 'e')) {  // " 're", " 've"
                            total--;  // remove space
                        } else {
                            //total--;  // remove space
                        }
                    } else {
                        //total--;  // remove space
                    }
                }
            }
            text[total++] = x;
        }
    }

    return total <= text_len_max ? total : -total;
}

std::string llama_token_to_piece(const struct llama_vocab & vocab, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache
    const int n_chars = llama_token_to_piece_impl(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece_impl(vocab, token, &piece[0], piece.size(), 0, special);
        assert(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

void llm_load_vocab(llama_vocab & vocab, struct gguf_context * ctx) {
    llm_arch arch = LLM_ARCH_UNKNOWN;
    LLM_KV kv = LLM_KV(arch);

    auto key_id = gguf_find_key(ctx, kv(LLM_KV_GENERAL_ARCHITECTURE).c_str());
    if (key_id != -1) {
        std::string arch_name = gguf_get_val_str(ctx, key_id);
        arch = llm_arch_from_string(arch_name);
        kv = LLM_KV(arch);
    }

    // determine vocab type
    {
        std::string tokenizer_model;
        std::string tokenizer_pre;

        auto key_id = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_MODEL).c_str());
        tokenizer_model = gguf_get_val_str(ctx, key_id);

        key_id = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_PRE).c_str());
        if (key_id != -1) {
            tokenizer_pre = gguf_get_val_str(ctx, key_id);
        }

        if (tokenizer_model == "no_vocab") {
            vocab.type = LLAMA_VOCAB_TYPE_NONE;

            // default special tokens
            vocab.special_bos_id  = -1;
            vocab.special_eos_id  = -1;
            vocab.special_unk_id  = -1;
            vocab.special_sep_id  = -1;
            vocab.special_pad_id  = -1;
            vocab.special_cls_id  = -1;
            vocab.special_mask_id = -1;
            vocab.linefeed_id     = -1;

            // read vocab size from metadata
            key_id = gguf_find_key(ctx, kv(LLM_KV_VOCAB_SIZE).c_str());
            vocab.n_vocab = gguf_get_val_u32(ctx, key_id);
            return;
        }

        if (tokenizer_model == "llama") {
            vocab.type = LLAMA_VOCAB_TYPE_SPM;

            // default special tokens
            vocab.special_bos_id  = 1;
            vocab.special_eos_id  = 2;
            vocab.special_unk_id  = 0;
            vocab.special_sep_id  = -1;
            vocab.special_pad_id  = -1;
            vocab.special_cls_id  = -1;
            vocab.special_mask_id = -1;
        } else if (tokenizer_model == "bert") {
            vocab.type = LLAMA_VOCAB_TYPE_WPM;

            // default special tokens
            vocab.special_bos_id  = -1;
            vocab.special_eos_id  = -1;
            vocab.special_unk_id  = 100;
            vocab.special_sep_id  = 102;
            vocab.special_pad_id  = 0;
            vocab.special_cls_id  = 101;
            vocab.special_mask_id = 103;
        } else if (tokenizer_model == "gpt2") {
            vocab.type = LLAMA_VOCAB_TYPE_BPE;

            // read bpe merges and populate bpe ranks
            const int merges_keyidx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_MERGES).c_str());
            if (merges_keyidx == -1) {
                throw std::runtime_error("cannot find tokenizer merges in model file\n");
            }

            const int n_merges = gguf_get_arr_n(ctx, merges_keyidx);
            for (int i = 0; i < n_merges; i++) {
                const std::string word = gguf_get_arr_str(ctx, merges_keyidx, i);
                GGML_ASSERT(unicode_cpts_from_utf8(word).size() > 0);

                std::string first;
                std::string second;

                const size_t pos = word.find(' ', 1);

                if (pos != std::string::npos) {
                    first  = word.substr(0, pos);
                    second = word.substr(pos + 1);
                }

                vocab.bpe_ranks.emplace(std::make_pair(first, second), i);
            }

            // default special tokens
            vocab.special_bos_id  = 11;
            vocab.special_eos_id  = 11;
            vocab.special_unk_id  = -1;
            vocab.special_sep_id  = -1;
            vocab.special_pad_id  = -1;
            vocab.special_cls_id  = -1;
            vocab.special_mask_id = -1;
        } else if (tokenizer_model == "t5") {
            vocab.type = LLAMA_VOCAB_TYPE_UGM;

            // default special tokens
            vocab.special_bos_id  = -1;
            vocab.special_eos_id  = 1;
            vocab.special_unk_id  = 2;
            vocab.special_sep_id  = -1;
            vocab.special_pad_id  = 0;
            vocab.special_cls_id  = -1;
            vocab.special_mask_id = -1;

            const int precompiled_charsmap_keyidx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_PRECOMPILED_CHARSMAP).c_str());
            if (precompiled_charsmap_keyidx != -1) {
                size_t n_precompiled_charsmap = gguf_get_arr_n(ctx, precompiled_charsmap_keyidx);
                const char * precompiled_charsmap = (const char *) gguf_get_arr_data(ctx, precompiled_charsmap_keyidx);
                vocab.precompiled_charsmap.assign(precompiled_charsmap, precompiled_charsmap + n_precompiled_charsmap);
#ifdef IS_BIG_ENDIAN
                // correct endiannes of data in precompiled_charsmap binary blob
                uint32_t * xcda_blob_size = (uint32_t *) &vocab.precompiled_charsmap[0];
                *xcda_blob_size = __builtin_bswap32(*xcda_blob_size);
                assert(*xcda_blob_size + sizeof(uint32_t) < n_precompiled_charsmap);
                size_t xcda_array_size = *xcda_blob_size / sizeof(uint32_t);
                uint32_t * xcda_array = (uint32_t *) &vocab.precompiled_charsmap[sizeof(uint32_t)];
                for (size_t i = 0; i < xcda_array_size; ++i) {
                    xcda_array[i] = __builtin_bswap32(xcda_array[i]);
                }
#endif
            }
        } else if (tokenizer_model == "rwkv") {
            vocab.type = LLAMA_VOCAB_TYPE_RWKV;

            // default special tokens
            vocab.special_bos_id = -1;
            vocab.special_eos_id = -1;
            vocab.special_unk_id = -1;
            vocab.special_sep_id = -1;
            vocab.special_pad_id = -1;
        } else {
            throw std::runtime_error(format("unknown tokenizer: '%s'", tokenizer_model.c_str()));
        }

        // for now, only BPE models have pre-tokenizers
        if (vocab.type == LLAMA_VOCAB_TYPE_BPE) {
            vocab.tokenizer_add_space_prefix = false;
            vocab.tokenizer_clean_spaces = true;
            if (tokenizer_pre.empty()) {
                fprintf(stderr, "%s: missing pre-tokenizer type, using: 'default'\n", __func__);
                fprintf(stderr, "%s:                                             \n", __func__);
                fprintf(stderr, "%s: ************************************        \n", __func__);
                fprintf(stderr, "%s: GENERATION QUALITY WILL BE DEGRADED!        \n", __func__);
                fprintf(stderr, "%s: CONSIDER REGENERATING THE MODEL             \n", __func__);
                fprintf(stderr, "%s: ************************************        \n", __func__);
                fprintf(stderr, "%s:                                             \n", __func__);
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            } else if (tokenizer_pre == "default") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            } else if (
                    tokenizer_pre == "llama3"   ||
                    tokenizer_pre == "llama-v3" ||
                    tokenizer_pre == "llama-bpe") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_LLAMA3;
                vocab.tokenizer_ignore_merges = true;
                vocab.tokenizer_add_bos = true;
            } else if (
                    tokenizer_pre == "deepseek-llm") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                    tokenizer_pre == "deepseek-coder") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                    tokenizer_pre == "falcon") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_FALCON;
            } else if (
                    tokenizer_pre == "mpt") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_MPT;
            } else if (
                    tokenizer_pre == "starcoder") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_STARCODER;
            } else if (
                    tokenizer_pre == "gpt-2"   ||
                    tokenizer_pre == "phi-2"   ||
                    tokenizer_pre == "jina-es" ||
                    tokenizer_pre == "jina-de" ||
                    tokenizer_pre == "jina-v2-es" ||
                    tokenizer_pre == "jina-v2-de" ||
                    tokenizer_pre == "jina-v2-code") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_GPT2;
            } else if (
                    tokenizer_pre == "refact") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_REFACT;
            } else if (
                tokenizer_pre == "command-r") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_COMMAND_R;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                tokenizer_pre == "qwen2") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_QWEN2;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                tokenizer_pre == "stablelm2") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_STABLELM2;
            } else if (
                tokenizer_pre == "olmo") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_OLMO;
            } else if (
                tokenizer_pre == "dbrx") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DBRX;
            } else if (
                tokenizer_pre == "smaug-bpe") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_SMAUG;
            } else if (
                tokenizer_pre == "poro-chat") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_PORO;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                tokenizer_pre == "chatglm-bpe") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_CHATGLM4;
                vocab.special_bos_id  = -1;
            } else if (
                tokenizer_pre == "viking") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_VIKING;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                tokenizer_pre == "jais") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_JAIS;
            } else if (
                tokenizer_pre == "tekken") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_TEKKEN;
                vocab.tokenizer_clean_spaces = false;
                vocab.tokenizer_ignore_merges = true;
                vocab.tokenizer_add_bos = true;
            } else if (
                tokenizer_pre == "smollm") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_SMOLLM;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                tokenizer_pre == "codeshell") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_CODESHELL;
            } else if (
                tokenizer_pre == "bloom") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_BLOOM;
            } else if (
                tokenizer_pre == "gpt3-finnish") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH;
            } else if (
                tokenizer_pre == "exaone") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_EXAONE;
            } else {
                throw std::runtime_error(format("unknown pre-tokenizer type: '%s'", tokenizer_pre.c_str()));
            }
        } else if (vocab.type == LLAMA_VOCAB_TYPE_SPM) {
            vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            vocab.tokenizer_add_space_prefix = true;
            vocab.tokenizer_clean_spaces = false;
            vocab.tokenizer_add_bos = true;
            vocab.tokenizer_add_eos = false;
        } else if (vocab.type == LLAMA_VOCAB_TYPE_WPM) {
            vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            vocab.tokenizer_add_space_prefix = false;
            vocab.tokenizer_clean_spaces = true;
            vocab.tokenizer_add_bos = true;
            vocab.tokenizer_add_eos = false;
        } else if (vocab.type == LLAMA_VOCAB_TYPE_UGM) {
            vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            vocab.tokenizer_add_bos = false;
            vocab.tokenizer_add_eos = true;
        } else if (vocab.type == LLAMA_VOCAB_TYPE_RWKV) {
            vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            vocab.tokenizer_add_space_prefix = false;
            vocab.tokenizer_clean_spaces = false;
            vocab.tokenizer_add_bos = false;
            vocab.tokenizer_add_eos = false;
        } else {
            vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
        }

        key_id = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_ADD_PREFIX).c_str());
        if (key_id != -1) {
            vocab.tokenizer_add_space_prefix = gguf_get_val_bool(ctx, key_id);
        }

        key_id = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_REMOVE_EXTRA_WS).c_str());
        if (key_id != -1) {
            vocab.tokenizer_remove_extra_whitespaces = gguf_get_val_bool(ctx, key_id);
        }
    }

    const int token_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_LIST).c_str());
    if (token_idx == -1) {
        throw std::runtime_error("cannot find tokenizer vocab in model file\n");
    }

    const float * scores = nullptr;
    const int score_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_SCORES).c_str());
    if (score_idx != -1) {
        scores = (const float * ) gguf_get_arr_data(ctx, score_idx);
    }

    const int * toktypes = nullptr;
    const int toktype_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE).c_str());
    if (toktype_idx != -1) {
        toktypes = (const int * ) gguf_get_arr_data(ctx, toktype_idx);
    }

    const uint32_t n_vocab = gguf_get_arr_n(ctx, token_idx);

    vocab.n_vocab = n_vocab;
    vocab.id_to_token.resize(n_vocab);

    for (uint32_t i = 0; i < n_vocab; i++) {
        std::string word = gguf_get_arr_str(ctx, token_idx, i);
        //GGML_ASSERT(unicode_cpts_from_utf8(word).size() > 0);

        vocab.token_to_id[word] = i;
        vocab.max_token_len = std::max(vocab.max_token_len, (int) word.size());

        auto & token_data = vocab.id_to_token[i];
        token_data.text  = std::move(word);
        token_data.score = scores ? scores[i] : 0.0f;
        token_data.attr  = LLAMA_TOKEN_ATTR_NORMAL;

        if (toktypes) {  //TODO: remove, required until per token attributes are available from GGUF file
            switch(toktypes[i]) {
                case LLAMA_TOKEN_TYPE_UNKNOWN:      token_data.attr = LLAMA_TOKEN_ATTR_UNKNOWN;      break;
                case LLAMA_TOKEN_TYPE_UNUSED:       token_data.attr = LLAMA_TOKEN_ATTR_UNUSED;       break;
                case LLAMA_TOKEN_TYPE_NORMAL:       token_data.attr = LLAMA_TOKEN_ATTR_NORMAL;       break;
                case LLAMA_TOKEN_TYPE_CONTROL:      token_data.attr = LLAMA_TOKEN_ATTR_CONTROL;      break;
                case LLAMA_TOKEN_TYPE_USER_DEFINED: token_data.attr = LLAMA_TOKEN_ATTR_USER_DEFINED; break;
                case LLAMA_TOKEN_TYPE_BYTE:         token_data.attr = LLAMA_TOKEN_ATTR_BYTE;         break;
                case LLAMA_TOKEN_TYPE_UNDEFINED:    token_data.attr = LLAMA_TOKEN_ATTR_UNDEFINED;    break;
                default:                            token_data.attr = LLAMA_TOKEN_ATTR_UNDEFINED;    break;
            }
        }
    }
    GGML_ASSERT(vocab.id_to_token.size() == vocab.token_to_id.size());

    // determine the newline token: LLaMA "<0x0A>" == 10 == '\n', Falcon 193 == '\n'
    if (vocab.type == LLAMA_VOCAB_TYPE_SPM) {
        // For Fill-In-the-Middle (FIM)/infill models which where converted
        // prior to support of FIM special tokens in GGUF, the following
        // will allow those models to continue to work. The general names
        // of the known models are currently CodeLlama (LLM_ARCH_LLAMA) and
        // CodeGemma (LLM_ARCH_GEMMA). This can potentially be removed once
        // new versions of these models have been published.
        std::string gen_name;

        auto key_id = gguf_find_key(ctx, kv(LLM_KV_GENERAL_NAME).c_str());
        if (key_id != -1) {
            gen_name = gguf_get_val_str(ctx, key_id);
        }

        std::transform(gen_name.begin(), gen_name.end(), gen_name.begin(),
            [](unsigned char c){ return std::tolower(c); });

        if (gen_name.find("code") != std::string::npos) {
            if (arch == LLM_ARCH_LLAMA
              && 32010 < vocab.id_to_token.size()
              && vocab.id_to_token[32007].text.find("<PRE>") != std::string::npos
              && vocab.id_to_token[32008].text.find("<SUF>") != std::string::npos
              && vocab.id_to_token[32009].text.find("<MID>") != std::string::npos
              && vocab.id_to_token[32010].text.find("<EOT>") != std::string::npos) {
                vocab.special_prefix_id = 32007;
                vocab.special_suffix_id = 32008;
                vocab.special_middle_id = 32009;
                vocab.special_eot_id    = 32010;
            } else if (arch == LLM_ARCH_GEMMA
              && 107 < vocab.id_to_token.size()
              && vocab.id_to_token[67].text == "<|fim_prefix|>"
              && vocab.id_to_token[69].text == "<|fim_suffix|>"
              && vocab.id_to_token[68].text == "<|fim_middle|>"
              && vocab.id_to_token[107].text == "<end_of_turn>") {
                vocab.special_prefix_id = 67;
                vocab.special_suffix_id = 69;
                vocab.special_middle_id = 68;
                // TODO: this is not EOT, it is "file separator" token, needs fix
                //       https://huggingface.co/google/codegemma-7b-it/blob/9b1d9231388358c04d90bd003458f5070d97db44/tokenizer_config.json#L565-L572
                //vocab.special_eot_id    = 70;
                vocab.special_eot_id    = 107;
            }
        }
        try {
            vocab.linefeed_id = llama_byte_to_token_impl(vocab, '\n');
        } catch (const std::exception & e) {
            fprintf(stderr, "%s: SPM vocabulary, but newline token not found: %s! Using special_pad_id instead.", __func__, e.what());
            vocab.linefeed_id = vocab.special_pad_id;
        }
    } else if (vocab.type == LLAMA_VOCAB_TYPE_WPM) {
        vocab.linefeed_id = vocab.special_pad_id;
    } else if (vocab.type == LLAMA_VOCAB_TYPE_RWKV) {
        const std::vector<int> ids = llama_tokenize_internal(vocab, "\n", false);
        GGML_ASSERT(!ids.empty() && "model vocab missing newline token");
        vocab.linefeed_id = ids[0];
    } else {
        const std::vector<int> ids = llama_tokenize_internal(vocab, "\xC4\x8A", false); // U+010A
        GGML_ASSERT(!ids.empty() && "model vocab missing newline token");
        vocab.linefeed_id = ids[0];
    }

    // special tokens
    {
        const std::vector<std::pair<enum llm_kv, int32_t &>> special_token_types = {
            { LLM_KV_TOKENIZER_BOS_ID,    vocab.special_bos_id    },
            { LLM_KV_TOKENIZER_EOS_ID,    vocab.special_eos_id    },
            { LLM_KV_TOKENIZER_UNK_ID,    vocab.special_unk_id    },
            { LLM_KV_TOKENIZER_SEP_ID,    vocab.special_sep_id    },
            { LLM_KV_TOKENIZER_PAD_ID,    vocab.special_pad_id    },
            { LLM_KV_TOKENIZER_CLS_ID,    vocab.special_cls_id    },
            { LLM_KV_TOKENIZER_MASK_ID,   vocab.special_mask_id   },
            { LLM_KV_TOKENIZER_PREFIX_ID, vocab.special_prefix_id },
            { LLM_KV_TOKENIZER_SUFFIX_ID, vocab.special_suffix_id },
            { LLM_KV_TOKENIZER_MIDDLE_ID, vocab.special_middle_id },
            { LLM_KV_TOKENIZER_EOT_ID,    vocab.special_eot_id    },
            { LLM_KV_TOKENIZER_EOM_ID,    vocab.special_eom_id    },
        };

        for (const auto & it : special_token_types) {
            const std::string & key = kv(std::get<0>(it));
            int32_t & id = std::get<1>(it);

            auto key_id = gguf_find_key(ctx, key.c_str());
            if (key_id == -1) {
                continue;
            }

            uint32_t new_id = gguf_get_val_u32(ctx, key_id);
            if (new_id >= vocab.id_to_token.size()) {
                fprintf(stderr, "%s: bad special token: '%s' = %ud, using default id %d\n",
                    __func__, key.c_str(), new_id, id);
            } else {
                id = new_id;
            }
        }

        // Handle add_bos_token and add_eos_token
        {
            auto key_id = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_ADD_BOS).c_str());
            if (key_id != -1) {
                vocab.tokenizer_add_bos = gguf_get_val_bool(ctx, key_id);
            }

            key_id = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_ADD_EOS).c_str());
            if (key_id != -1) {
                vocab.tokenizer_add_eos = gguf_get_val_bool(ctx, key_id);
            }
        }

        // find EOT token: "<|eot_id|>", "<|im_end|>", "<end_of_turn>", etc.
        //
        // TODO: convert scripts should provide this token through the KV metadata LLAMA_KV_TOKENIZER_EOT_ID
        //       for now, we apply this workaround to find the EOT token based on its text
        if (vocab.special_eot_id == -1) {
            for (const auto & t : vocab.token_to_id) {
                if (
                        // TODO: gemma "<end_of_turn>" is exported as a normal token, so the following check does not work
                        //       need to fix convert script
                        //vocab.id_to_token[t.second].type == LLAMA_TOKEN_TYPE_CONTROL &&
                        (t.first == "<|eot_id|>" ||
                         t.first == "<|im_end|>" ||
                         t.first == "<|end|>" ||
                         t.first == "<end_of_turn>" ||
                         t.first == "<|endoftext|>"
                        )
                   ) {
                    vocab.special_eot_id = t.second;
                    if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        fprintf(stderr, "%s: control-looking token: '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                            __func__, t.first.c_str());
                        vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                    break;
                }
            }
        }

        // find EOM token: "<|eom_id|>"
        //
        // TODO: convert scripts should provide this token through the KV metadata LLAMA_KV_TOKENIZER_EOM_ID
        //       for now, we apply this workaround to find the EOM token based on its text
        if (vocab.special_eom_id == -1) {
            const auto & t = vocab.token_to_id.find("<|eom_id|>");
            if (t != vocab.token_to_id.end()) {
                vocab.special_eom_id = t->second;
                if ((vocab.id_to_token[t->second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                    fprintf(stderr, "%s: control-looking token: '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                        __func__, t->first.c_str());
                    vocab.id_to_token[t->second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                }
            }
        }
    }

    // build special tokens cache
    {
        for (llama_vocab::id id = 0; id < (llama_vocab::id)n_vocab; ++id) {
            if (vocab.id_to_token[id].attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN)) {
                vocab.cache_special_tokens.push_back(id);
            }
        }

        std::sort(vocab.cache_special_tokens.begin(), vocab.cache_special_tokens.end(),
            [&] (const llama_vocab::id a, const llama_vocab::id b) {
                return vocab.id_to_token[a].text.size() > vocab.id_to_token[b].text.size();
            }
        );

        // printf("%s: special tokens cache size = %u\n", __func__, (uint32_t)vocab.cache_special_tokens.size());
    }

    // build token to piece cache
    {
        size_t size_cache = 0;

        std::vector<llama_vocab::token> cache_token_to_piece(n_vocab);

        for (uint32_t id = 0; id < n_vocab; ++id) {
            cache_token_to_piece[id] = llama_token_to_piece(vocab, id, true);

            size_cache += cache_token_to_piece[id].size();
        }

        std::swap(vocab.cache_token_to_piece, cache_token_to_piece);

        // printf("%s: token to piece cache size = %.4f MB\n", __func__, size_cache / 1024.0 / 1024.0);
    }

    // Handle per token attributes
    //NOTE: Each model customizes per token attributes.
    //NOTE: Per token attributes are missing from the GGUF file.
    //TODO: Extract attributes from GGUF file.
    {
        auto _contains_any = [] (const std::string &str, const std::vector<std::string> &substrs) -> bool {
            for (auto substr : substrs) {
                if (str.find(substr) < std::string::npos) {
                    return true;
                }
            }
            return false;
        };

        auto _set_tokenid_attr = [&] (const llama_vocab::id id, llama_token_attr attr, bool value) {
            uint32_t current = vocab.id_to_token.at(id).attr;
            current = value ? (current | attr) : (current & ~attr);
            vocab.id_to_token[id].attr = (llama_token_attr) current;
        };

        auto _set_token_attr = [&] (const std::string & token, llama_token_attr attr, bool value) {
            _set_tokenid_attr(vocab.token_to_id.at(token), attr, value);
        };


        std::string tokenizer_pre;
        auto key_id = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_PRE).c_str());
        if (key_id != -1) {
            tokenizer_pre = gguf_get_val_str(ctx, key_id);
        }

        std::string model_name;
        key_id = gguf_find_key(ctx, kv(LLM_KV_GENERAL_NAME).c_str());
        if (key_id != -1) {
            model_name = gguf_get_val_str(ctx, key_id);
        }

        // model name to lowercase
        std::transform(model_name.begin(), model_name.end(), model_name.begin(),
            [] (const std::string::value_type x) {
                return std::tolower(x);
            }
        );

        // set attributes by model/tokenizer name
        if (_contains_any(tokenizer_pre, {"jina-v2-de", "jina-v2-es", "jina-v2-code"})) {
            _set_token_attr("<mask>", LLAMA_TOKEN_ATTR_LSTRIP, true);
        } else if (_contains_any(model_name, {"phi-3", "phi3"})) {
            for (auto id : vocab.cache_special_tokens) {
                _set_tokenid_attr(id, LLAMA_TOKEN_ATTR_RSTRIP, true);
            }
            for (auto token : {"</s>"}) {
                _set_token_attr(token, LLAMA_TOKEN_ATTR_RSTRIP, true);
            }
            for (auto token : {"<unk>", "<s>", "<|endoftext|>"}) {
                _set_token_attr(token, LLAMA_TOKEN_ATTR_RSTRIP, false);
            }
        }
    }
}
