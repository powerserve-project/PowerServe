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

#include "core/config.hpp"
#include "speculative/speculative_config.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace powerserve {

struct CommandLineArgument {
    /*
     * Environment Configuration
     */

    /// The folder containing workspace configuration and models
    std::string work_folder;

    /// The folder containing qnn libraries, use work_folder/qnn_libs as default if not set.
    std::string qnn_lib_folder;

    /// The path to the workspace configuration
    std::string workspace_config_path;

    /// The number of thread for inference
    uint32_t num_thread = 0;

    /*
     * Model & Input & Output Configuration
     */

    /// The model name of main model
    std::string main_model;

    /// The model name of dragt model (only for speculative inference)
    std::string draft_model;

    /// Input prompt string
    std::string prompt = "Once upon a time";

    /// The maximum length of output token list
    uint32_t num_predict = 16;

    /// The maximum number of tokens processed in one iteration
    uint32_t batch_size = 0;

    /// Disable QNN backend
    bool no_qnn = false;

    /*
     * Speculative
     */
    bool use_spec = false;

    SpeculativeConfig speculative_config;

    /*
     * Server Configuration
     */

    /// The ip address the server to listen on
    std::string host = "127.0.0.1";

    /// The ip port the server to listen on
    uint32_t port = 8080;
};

/*!
 * @brief Parse the command arguments from the program input.
 */
CommandLineArgument parse_command_line(const std::string_view program_name, int argc, char **argv);

/*!
 * @brief Parse and generate config according to the command line arguments.
 * @param[in] args The command line argument to overwrite config
 */
Config get_config_from_argument(const CommandLineArgument &args);

} // namespace powerserve
