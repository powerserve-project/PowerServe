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

#include "cmdline.hpp"

#include <exception>

#ifdef POWERSERVE_WITH_QNN
#include "backend/qnn/config.hpp"
#endif // POWERSERVE_WITH_QNN

#include "CLI/CLI.hpp"
#include "core/config.hpp"
#include "core/logger.hpp"
#include "core/typedefs.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace powerserve {

static bool read_prompt_from_file(const std::string &prompt_file, std::string &prompt) {
    std::ifstream file(prompt_file);
    if (!file.good()) {
        POWERSERVE_LOG_ERROR("failed to open prompt file: {}", prompt_file);
        return false;
    }

    std::stringstream prompt_stream;
    prompt_stream << file.rdbuf();
    prompt = prompt_stream.str();
    return true;
}

CommandLineArgument parse_command_line(const std::string_view program_name, int argc, char **argv) {
    CommandLineArgument args;

    powerserve::print_timestamp();
    CLI::App app(program_name.data());

    /*
     * Environment Configuration
     */

    app.add_option("-d,--work-folder", args.work_folder, "Set the working folder (required).")->required();
    app.add_option("--lib-folder", args.qnn_lib_folder, "Set the path to the directory of QNN libraries.");
    app.add_option("--workspace-config", args.workspace_config_path, "Set the path to the workspace config.");

    app.add_option("-t,--thread", args.num_thread, "Set the number of threads for inference.");
    app.add_option("-b,--batch_size", args.batch_size, "Set the number of batch size for prefill.");

    /*
     * Model Configuration
     */
    app.add_option("-m,--model", args.main_model, "Set the model name or path to the main model directory.");
    app.add_option("--draft-model", args.draft_model, "Set the model name or path to the draft model directory.");
#if defined(POWERSERVE_WITH_QNN)
    app.add_flag("--use-spec", args.use_spec, "Use QNN speculative decode.");
#endif

    auto &speculative_config = args.speculative_config;
    app.add_option("--draft-batch-size", speculative_config.draft_batch_size)->capture_default_str();
    app.add_option("--draft-sampler-top-k", speculative_config.draft_sampler.top_k)->capture_default_str();
    app.add_option("--draft-sampler-temperature", speculative_config.draft_sampler.temperature)->capture_default_str();
    app.add_option("--draft-sampler-p-base", speculative_config.draft_sampler.p_base)->capture_default_str();
    app.add_option("--token-tree-max-fan-out", speculative_config.token_tree.max_fan_out)->capture_default_str();
    app.add_option("--token-tree-min-prob", speculative_config.token_tree.min_prob)->capture_default_str();
    app.add_option("--token-tree-early-stop", speculative_config.token_tree.early_stop)->capture_default_str();
    /*
     * Input Configuration
     */

    { // prompt input
        CLI::Option_group *prompt_group =
            app.add_option_group("Prompt Options", "Choose either prompt or prompt-file, not both.");

        // input prompt string directly
        prompt_group->add_option("-p,--prompt", args.prompt, "Set the prompt string.");

        // input file content as input prompt
        prompt_group->add_option(
            "-f,--prompt-file",
            [&args](const std::vector<std::string> &argument) {
                if (argument.size() != 1) {
                    return false;
                }
                const std::string &prompt_file = argument[0];
                return read_prompt_from_file(prompt_file, args.prompt);
            },
            "Set the prompt as the content of the file."
        );

        prompt_group->require_option(0, 1);
    }

    /*
     * Output Configuration
     */
    app.add_option("-n,--n-predicts", args.num_predict, "Specify the number of predictions to make.");

    /*
     * Backend Configuration
     */
    app.add_flag("--no-qnn", args.no_qnn, "Disable QNN processing.");

    /*
     * Server Configuration
     */
    app.add_option("--host", args.host, "Set the ip address the server to listen.");
    app.add_option("--port", args.port, "Set the ip address the server to listen.");

    /*
     * Finalize
     */
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &err) {
        const int ret = app.exit(err);
        exit(ret);
    }

    /*
     * Post Processing
     * Setting default value for non-specified values
     */

    if (args.workspace_config_path.empty()) {
        args.workspace_config_path = powerserve::Path(args.work_folder) / powerserve::WORKSPACE_CONFIG_FILENAME;
    }

#ifdef POWERSERVE_WITH_QNN
    if (args.qnn_lib_folder.empty()) {
        args.qnn_lib_folder = std::filesystem::path(args.work_folder) / powerserve::qnn::QNN_LIB_DIR_NAME;
    }
#endif // POWERSERVE_WITH_QNN

    return args;
}

Config get_config_from_argument(const CommandLineArgument &args) {
    Config config(args.work_folder, args.workspace_config_path);

    if (!args.main_model.empty()) {
        const Path inner_model_path = Path(args.work_folder) / args.main_model;
        if (std::filesystem::exists(inner_model_path)) {
            config.main_model_dir = inner_model_path;
        } else if (std::filesystem::exists(args.main_model)) {
            config.main_model_dir = args.main_model;
        }
    }

    if (!args.draft_model.empty()) {
        const Path inner_model_path = Path(args.work_folder) / args.draft_model;
        if (std::filesystem::exists(inner_model_path)) {
            config.draft_model_dir = inner_model_path;
        } else if (std::filesystem::exists(args.draft_model)) {
            config.draft_model_dir = args.draft_model;
        }
    }

    auto &hyper_params = config.hyper_params;

    if (args.num_thread != 0) {
        hyper_params.n_threads = args.num_thread;
    }

    if (args.batch_size != 0) {
        hyper_params.batch_size = args.batch_size;
    }

    // TODO: sampler config overwrite

    return config;
}

} // namespace powerserve
