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

#include "tokenizer.hpp"

#include "core/logger.hpp"
#include "ggml.h"

namespace powerserve {

Tokenizer::Tokenizer(const Path &vocab_path) {
    struct ggml_context *ctx       = nullptr;
    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx      = &ctx,
    };

    struct gguf_context *meta = gguf_init_from_file(vocab_path.c_str(), params);
    POWERSERVE_ASSERT(meta);

    const int template_key_id = gguf_find_key(meta, "tokenizer.chat_template");
    if (template_key_id != -1) {
        m_template_type = gguf_get_val_str(meta, template_key_id);
    } else {
        m_template_type = "chatml";
        POWERSERVE_LOG_ERROR(
            "failed to find kv entry <tokenizer.chat_template>, use chat template `{}` as default.", m_template_type
        );
    }

    llm_load_vocab(m_vocab, meta);

    gguf_free(meta);

    debug_tokenizer();
}

size_t Tokenizer::n_vocabs() const {
    return m_vocab.n_vocab;
}

auto Tokenizer::bos_token() const -> Token {
    return m_vocab.special_bos_id;
}

bool Tokenizer::should_stop(Token token) const {
    return token == m_vocab.special_bos_id || token == m_vocab.special_eom_id || token == m_vocab.special_eos_id ||
           token == m_vocab.special_eot_id;
}

auto Tokenizer::tokenize(const std::string &text, bool add_special) const -> std::vector<Token> {
    return llama_tokenize_internal(m_vocab, text, add_special, true);
}

auto Tokenizer::to_string(Token token, bool special) const -> std::string {
    return llama_token_to_piece(m_vocab, token, special);
}

/*!
 * @brief automatically search and apply chat template
 * @ref llama.cpp
 */
static std::string apply_chat_template_internal(
    const std::string &template_type, const std::vector<ChatEntry> &chat, const bool add_ass
) {
    std::stringstream ss;

    const auto template_contains = [&template_type](std::string haystack) -> bool {
        return template_type.find(haystack) != std::string::npos;
    };

    if (template_type == "chatml" || template_contains("<|im_start|>")) {
        // chatml template
        for (const auto &message : chat) {
            ss << fmt::format("<|im_start|>{}\n{}<|im_end|>", message.role, message.content);
        }
        if (add_ass) {
            ss << "<|im_start|>assistant\n";
        }
    } else if (template_type == "llama2" || template_type == "mistral" || template_contains("[INST]")) {
        // llama2 template and its variants
        // [variant] support system message
        const bool support_system_message = template_contains("<<SYS>>") || template_type == "mistral";
        // [variant] space before + after response
        const bool space_around_response = template_contains("' ' + eos_token");
        // [variant] add BOS inside history
        const bool add_bos_inside_history = template_contains("bos_token + '[INST]");
        // [variant] trim spaces from the input message
        const bool strip_message = template_contains("content.strip()");
        // construct the prompt
        bool is_inside_turn = true; // skip BOS at the beginning
        ss << "[INST] ";
        for (const auto &message : chat) {
            const std::string content = strip_message ? trim(message.content) : message.content;
            const std::string role    = message.role;

            if (!is_inside_turn) {
                is_inside_turn = true;
                ss << (add_bos_inside_history ? "<s>[INST] " : "[INST] ");
            }
            if (role == "system") {
                if (support_system_message) {
                    ss << fmt::format("<<SYS>>\n{}\n<</SYS>>\n\n", content);
                } else {
                    // if the model does not support system message, we still include it in the first message, but
                    // without <<SYS>>
                    ss << content << "\n";
                }
            } else if (role == "user") {
                ss << fmt::format("{} [/INST]", content);
            } else {
                const std::string_view around = (space_around_response ? " " : "");
                ss << fmt::format("{}{}{}</s>", around, content, around);
                is_inside_turn = false;
            }
        }
        // llama2 templates seem to not care about "add_generation_prompt"
    } else if (template_type == "phi3" || (template_contains("<|assistant|>") && template_contains("<|end|>"))) {
        // Phi 3
        for (const auto &message : chat) {
            ss << fmt::format("<|{}|>\n{}<|end|>\n", message.role, message.content);
        }
        if (add_ass) {
            ss << "<|assistant|>\n";
        }
    } else if (template_type == "zephyr" || template_contains("<|user|>")) {
        // zephyr template
        for (const auto &message : chat) {
            ss << fmt::format("<|{}|>\n{}<|endoftext|>\n", message.role, message.content);
        }
        if (add_ass) {
            ss << "<|assistant|>\n";
        }
    } else if (template_type == "monarch" || template_contains("bos_token + message['role']")) {
        // mlabonne/AlphaMonarch-7B template (the <s> is included inside history)
        for (bool first = false; const auto &message : chat) {
            std::string bos = "<s>"; // skip BOS for first message
            if (first) {
                bos   = "";
                first = true;
            }

            ss << fmt::format("{}{}\n{}</s>\n", bos, message.role, message.content);
        }
        if (add_ass) {
            ss << "<s>assistant\n";
        }
    } else if (template_type == "gemma" || template_type == "gemma2" || template_contains("<start_of_turn>")) {
        // google/gemma-7b-it
        std::string system_prompt = "";
        for (const auto &message : chat) {
            std::string role = message.role;
            if (role == "system") {
                // there is no system message for gemma, but we will merge it with user prompt, so nothing is broken
                system_prompt = trim(message.content);
                continue;
            }
            // in gemma, "assistant" is "model"
            role = role == "assistant" ? "model" : message.role;
            ss << "<start_of_turn>" << role << "\n";
            if (!system_prompt.empty() && role != "model") {
                ss << system_prompt << "\n\n";
                system_prompt = "";
            }
            ss << trim(message.content) << "<end_of_turn>\n";
        }
        if (add_ass) {
            ss << "<start_of_turn>model\n";
        }
    } else if (template_type == "orion" || template_contains("'\\n\\nAssistant: ' + eos_token")) {
        // OrionStarAI/Orion-14B-Chat
        std::string system_prompt = "";
        for (const auto &message : chat) {
            std::string role(message.role);
            if (role == "system") {
                // there is no system message support, we will merge it with user prompt
                system_prompt = message.content;
                continue;
            } else if (role == "user") {
                ss << "Human: ";
                if (!system_prompt.empty()) {
                    ss << system_prompt << "\n\n";
                    system_prompt = "";
                }
                ss << message.content << "\n\nAssistant: </s>";
            } else {
                ss << message.content << "</s>";
            }
        }
    } else if (template_type == "openchat" || template_contains("GPT4 Correct ")) {
        // openchat/openchat-3.5-0106,
        for (const auto &message : chat) {
            std::string role = message.role;
            if (role == "system") {
                ss << message.content << "<|end_of_turn|>";
            } else {
                role[0] = toupper(role[0]);
                ss << fmt::format("GPT4 Correct {}: {}<|end_of_turn|>", role, message.content);
            }
        }
        if (add_ass) {
            ss << "GPT4 Correct Assistant:";
        }
    } else if (template_type == "vicuna" || template_type == "vicuna-orca" ||
               (template_contains("USER: ") && template_contains("ASSISTANT: "))) {
        // eachadea/vicuna-13b-1.1 (and Orca variant)
        for (const auto &message : chat) {
            std::string role(message.role);
            if (role == "system") {
                // Orca-Vicuna variant uses a system prefix
                if (template_type == "vicuna-orca" || template_contains("SYSTEM: ")) {
                    ss << fmt::format("SYSTEM: {}\n", message.content);
                } else {
                    ss << message.content << "\n\n";
                }
            } else if (role == "user") {
                ss << fmt::format("USER: {}\n", message.content);
            } else if (role == "assistant") {
                ss << fmt::format("ASSISTANT: {}</s>\n", message.content);
            }
        }
        if (add_ass) {
            ss << "ASSISTANT:";
        }
    } else if (template_type == "deepseek" || (template_contains("### Instruction:") && template_contains("<|EOT|>"))) {
        // deepseek-ai/deepseek-coder-33b-instruct
        for (const auto &message : chat) {
            const std::string &role = message.role;
            if (role == "system") {
                ss << message.content;
            } else if (role == "user") {
                ss << fmt::format("### Instruction:\n{}\n", message.content);
            } else if (role == "assistant") {
                ss << fmt::format("### Response:\n{}\n<|EOT|>\n", message.content);
            }
        }
        if (add_ass) {
            ss << "### Response:\n";
        }
    } else if (template_type == "command-r" ||
               (template_contains("<|START_OF_TURN_TOKEN|>") && template_contains("<|USER_TOKEN|>"))) {
        // CohereForAI/c4ai-command-r-plus
        for (const auto &message : chat) {
            std::string role(message.role);
            if (role == "system") {
                ss << fmt::format(
                    "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{}<|END_OF_TURN_TOKEN|>", trim(message.content)
                );
            } else if (role == "user") {
                ss << fmt::format(
                    "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{}<|END_OF_TURN_TOKEN|>", trim(message.content)
                );
            } else if (role == "assistant") {
                ss << fmt::format(
                    "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{}<|END_OF_TURN_TOKEN|>", trim(message.content)
                );
                ;
            }
        }
        if (add_ass) {
            ss << "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>";
        }
    } else if (template_type == "llama3" ||
               (template_contains("<|start_header_id|>") && template_contains("<|end_header_id|>"))) {
        // Llama 3
        for (const auto &message : chat) {
            ss << fmt::format(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>", message.role, trim(message.content)
            );
        }
        if (add_ass) {
            ss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
        }
    } else if (template_type == "chatglm3" || template_type == "chatglm4" || template_contains("[gMASK]<sop>")) {
        ss << "[gMASK]<sop>";
        for (const auto &message : chat) {
            ss << fmt::format("<|{}|>\n{}", message.role, message.content);
        }
        if (add_ass) {
            ss << "<|assistant|>";
        }
    } else if (template_type == "minicpm" || template_contains((const char *)u8"<用户>")) {
        // MiniCPM-3B-OpenHermes-2.5-v2-GGUF
        for (const auto &message : chat) {
            std::string role(message.role);
            if (role == "user") {
                ss << fmt::format("{}{}<AI>", (const char *)u8"<用户>", trim(message.content));
            } else {
                ss << trim(message.content);
            }
        }
    } else if (template_type == "deepseek2" || template_contains("'Assistant: ' + message['content'] + eos_token")) {
        // DeepSeek-V2
        for (const auto &message : chat) {
            std::string role(message.role);
            if (role == "system") {
                ss << message.content << "\n\n";
            } else if (role == "user") {
                ss << fmt::format("User: {}\n\n", message.content);
            } else if (role == "assistant") {
                ss << fmt::format("Assistant: {}{}", message.content, (const char *)u8"<｜end▁of▁sentence｜>");
            }
        }
        if (add_ass) {
            ss << "Assistant:";
        }
    } else if (template_type == "exaone3" || (template_contains("[|system|]") && template_contains("[|assistant|]") &&
                                              template_contains("[|endofturn|]"))) {
        // ref: https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct/discussions/8#66bae61b1893d14ee8ed85bb
        // EXAONE-3.0-7.8B-Instruct
        for (const auto &message : chat) {
            std::string role(message.role);
            if (role == "system") {
                ss << fmt::format("[|system|]{}[|endofturn|]\n", trim(message.content));
            } else if (role == "user") {
                ss << fmt::format("[|user|]{}\n", trim(message.content));
            } else if (role == "assistant") {
                ss << fmt::format("[|assistant|]{}[|endofturn|]\n", trim(message.content));
            }
        }
        if (add_ass) {
            ss << "[|assistant|]";
        }
    } else {
        // template not supported
        POWERSERVE_LOG_ERROR("unknown template type: {}", template_type);
        return {};
    }
    return ss.str();
}

auto Tokenizer::apply_chat_template(const std::vector<ChatEntry> &chat_history, const bool add_ass) const
    -> std::string {
    // format the chat to string
    return apply_chat_template_internal(m_template_type, chat_history, add_ass);
}

} // namespace powerserve
