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

#include "CLI/CLI.hpp"
#include "backend/qnn/qnn.hpp"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

using namespace powerserve;

struct AutoTimeCounter {
public:
    std::string segment_name;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

public:
    AutoTimeCounter(std::string_view name) : segment_name(name) {
        start = std::chrono::steady_clock::now();
    }

    ~AutoTimeCounter() noexcept {
        end = std::chrono::steady_clock::now();
        print_timing();
    }

public:
    void print_timing() noexcept {
        const auto duration = end - start;
        std::cout << "[AutoTimeCounter] " << segment_name << " time: " << duration << std::endl;
    }
};

struct QNNContext {
public:
    std::unique_ptr<qnn::Session> session;
    std::unique_ptr<qnn::ContextBinary> context_binary;

public:
    QNNContext() = default;

    ~QNNContext() noexcept = default;
};

struct QNNAnonymousGraph {
private:
    QNNContext &context_ref_;

    qnn::Graph graph_;

    qnn::Tensor *input_;

    std::shared_ptr<qnn::SharedBuffer> input_buffer_;

    qnn::Tensor *output_;

    std::shared_ptr<qnn::SharedBuffer> output_buffer_;

public:
    QNNAnonymousGraph(QNNContext &context, std::string graph_name, const int graph_id) :
        context_ref_(context),
        graph_(*context_ref_.context_binary->m_context, graph_name) {
        graph_.set_n_hvx_threads(4);

        const std::string input_name  = "input_" + std::to_string(graph_id);
        const std::string output_name = "output_" + std::to_string(graph_id);
        input_                        = graph_.get_tensor(input_name, true);
        output_                       = graph_.get_tensor(output_name, true);
    }

    ~QNNAnonymousGraph() noexcept = default;

public:
    void execute() {
        graph_.execute();
    }

    size_t get_io_tensor_size() const {
        return input_->size() + output_->size();
    }

    void set_buffer() {
        input_buffer_ = std::make_shared<qnn::SharedBuffer>(
            *context_ref_.context_binary->m_context,
            *context_ref_.context_binary->m_alloc,
            input_->type(),
            input_->n_elements()
        );
        output_buffer_ = std::make_shared<qnn::SharedBuffer>(
            *context_ref_.context_binary->m_context,
            *context_ref_.context_binary->m_alloc,
            output_->type(),
            output_->n_elements()
        );

        input_->setup_shared_buffer(*input_buffer_);
        output_->setup_shared_buffer(*output_buffer_);
    }
};

int main(int argc, char **argv) {
    /*
     * Argument Parse
     */
    std::string qnn_path_str = "unknown";
    std::string model_name   = "simple_model";
    int num_graph            = 1;
    int repeat_times         = 1;

    CLI::App app("Simple test for QNN");

    app.add_option("--qnn-path", qnn_path_str, "The path to the QNN workspace")->required();
    app.add_option("--model-name", model_name, "The name of model");
    app.add_option("--graph-num", num_graph, "The number of graphs");
    app.add_option("--repeat", repeat_times, "The times to repeatedly execute QNN graph");
    CLI11_PARSE(app, argc, argv);

    std::filesystem::path qnn_path(qnn_path_str);

    std::string model_binary       = model_name + ".bin";
    std::filesystem::path bin_path = qnn_path / model_binary;

    QNNContext qnn_context;
    std::vector<QNNAnonymousGraph> graph_vec;
    graph_vec.reserve(num_graph);

    /*
     * Init backend and load necessary libraries
     */
    {
        AutoTimeCounter time_counter("Backend Initialization");

        qnn_context.session = std::make_unique<qnn::Session>(qnn_path);
    }

    /*
     * Load graph binary
     */
    {
        AutoTimeCounter time_counter("Graph Loading");

        qnn_context.context_binary = std::make_unique<qnn::ContextBinary>(*qnn_context.session->m_backend, bin_path);
        for (int graph_id = 0; graph_id < num_graph; ++graph_id) {
            // Create graph
            const std::string graph_name = model_name + '_' + std::to_string(graph_id);
            graph_vec.emplace_back(qnn_context, graph_name, graph_id);
        }

        size_t total_io_tensor_size = 0;
        for (const auto &graph : graph_vec) {
            total_io_tensor_size += graph.get_io_tensor_size();
        }
        qnn_context.context_binary->m_alloc = std::make_unique<qnn::SharedBufferAllocator>(total_io_tensor_size);

        for (auto &graph : graph_vec) {
            graph.set_buffer();
        }
    }

    /*
     * Execute graph
     */
    {
        AutoTimeCounter time_counter("Graph Warm Up");
        auto &qnn_graph = graph_vec[0];
        qnn_graph.execute();
    }
    {
        AutoTimeCounter time_counter("Graph Execution");

        for (int graph_id = 0; graph_id < num_graph; ++graph_id) {
            auto &qnn_graph = graph_vec[graph_id];
            // Execute graph
            for (int i = 0; i < repeat_times; ++i) {
                qnn_graph.execute();
            }
        }
    }

    /*
     * Summary
     */
    std::cout << "QNN Test Finished" << std::endl;
    return 0;
}
