import argparse
import json
import os
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F


HTP_SETTING_TEMPLATE = {
    "graphs": [{
        "graph_names": ["simple_model"],
        "fp16_relaxed_precision": 1,
        "vtcm_mb": 8,
        "O": 3,
        "hvx_threads": 4,
    }],
    "devices": [{
        "dsp_arch": "v79",
        "soc_model": 63,
    }],
    "context": {
        "weight_sharing_enabled": False,
    },
}

HTP_CONFIG_TEMPLATE = {
    "backend_extensions": {
        "shared_library_path": "libQnnHtpNetRunExtensions.so",
        "config_file_path": "htp_setting.json",
    },
    "context_configs": {
        "is_persistent_binary": True,
        "momory_limit_hint": 500000000,
        "enable_graphs": ["simple_model_0"],
    },
    "graph_configs": [{"graph_name": "simple_model_0"}],
}


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4096, 14336, bias=False)
        self.fc2 = nn.Linear(14336, 4096, bias=False)

    def forward(self, input):
        hidden_state = self.fc1(input)
        output = self.fc2(hidden_state)
        return output


def random_init_module(module):
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, 0.0, 1.0)


def run(cmd_args: list):
    cmd = " ".join(map(str, cmd_args))
    print(f"> {cmd}")
    ret = subprocess.Popen(cmd, shell=True).wait()
    assert ret == 0


def convert_model(qnn_sdk_root: str, model_name: str, model_id: int):
    """
    Convert the model to cpp
    """
    cmd_args = [
        f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-onnx-converter " "--input_network",
        f"{model_name}.onnx",
        "--input_dim",
        f"input_{model_id}",
        "32,4096",
        "--output_path",
        f"{model_name}.cpp",
    ]
    run(cmd_args)


def compile_model(qnn_sdk_root: str, model_name: str):
    """
    Convert the model to .so
    """
    cmd_args = [
        f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-model-lib-generator ",
        "-c",
        f"{model_name}.cpp",
        "-b",
        f"{model_name}.bin",
        "-o",
        "model_libs",
    ]
    run(cmd_args)


def convert_binary(qnn_sdk_root: str, model_name: str, model_num: int):
    """
    Convert the model to binaryt
    """

    model_name_list = [model_name + "_" + id.__str__() for id in range(model_num)]

    # Generate htp_config.json and htp_setting.json
    htp_config = HTP_CONFIG_TEMPLATE
    htp_config["context_configs"]["enable_graphs"] = model_name_list
    htp_config["graph_configs"] = [{"graph_name": cur_model_name} for cur_model_name in model_name_list]

    with open("htp_config.json", "w+") as f:
        json.dump(htp_config, f)

    htp_setting = HTP_SETTING_TEMPLATE
    htp_setting["graphs"][0]["graph_names"] = model_name_list
    with open("htp_setting.json", "w+") as f:
        json.dump(htp_setting, f)

    model_so_list = ""
    for cur_model_name in model_name_list:
        model_so_list += f"model_libs/x86_64-linux-clang/lib{cur_model_name}.so" + ","
    model_so_list.removesuffix(",")

    cmd_args = [
        f"{qnn_sdk_root}/bin/x86_64-linux-clang/qnn-context-binary-generator " "--backend",
        f"{qnn_sdk_root}/lib/x86_64-linux-clang/libQnnHtp.so",
        "--model",
        model_so_list,
        "--binary_file",
        model_name,
        "--config_file",
        "htp_config.json",
    ]
    run(cmd_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-num", type=int, default=1)
    parser.add_argument("--no-convert", action="store_true", help="Do not convert and compile model")
    args = parser.parse_args()

    qnn_sdk_root = os.getenv("QNN_SDK_ROOT")
    assert qnn_sdk_root is not None, "QNN_SDK_ROOT is not set"

    model_name = args.model_name
    model_num = args.model_num
    no_convert = args.no_convert

    if not no_convert:
        for model_id in range(model_num):
            cur_model_name = model_name + "_" + model_id.__str__()
            onnx_model_name = cur_model_name + ".onnx"

            torch_model = MLP()
            torch_model.apply(random_init_module)
            torch_input = torch.randn(32, 4096)
            torch.onnx.export(
                torch_model,
                torch_input,
                onnx_model_name,
                export_params=True,
                input_names=[f"input_{model_id}"],
                output_names=[f"output_{model_id}"],
            )

            convert_model(qnn_sdk_root, cur_model_name, model_id)
            compile_model(qnn_sdk_root, cur_model_name)

    convert_binary(qnn_sdk_root, model_name, model_num)
