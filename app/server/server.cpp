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
#include "simple_server.hpp"

#include <cstdlib>

int main(int argc, char *argv[]) {
    // 0. load config
    const powerserve::CommandLineArgument args = powerserve::parse_command_line("PowerServe CLI", argc, argv);

    simple_server_handler(args.work_folder, args.qnn_lib_folder, args.host, args.port);

    return 0;
}
