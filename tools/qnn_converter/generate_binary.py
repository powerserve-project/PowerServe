import argparse
import json
import os
import subprocess
from pathlib import Path

from soc_config import soc_map


parser = argparse.ArgumentParser()
parser.add_argument("--build-folder", type=Path, required=True)
parser.add_argument("--artifact-name", type=str, required=True)
parser.add_argument("--graph-names", type=str, nargs="+", required=True)
parser.add_argument("--soc", choices=soc_map.keys(), default="8650")
parser.add_argument("--log-file", type=str, default="build_bin.log")
parser.add_argument("--silent", action="store_true", help="Hide the shell command arguments.")

args = parser.parse_args()

build_folder: Path = args.build_folder

qnn_sdk_folder = os.getenv("QNN_SDK_ROOT")
assert qnn_sdk_folder is not None, "QNN_SDK_ROOT is not set"
qnn_sdk_folder = Path(qnn_sdk_folder)

graph_names = args.graph_names
lib_path = ""
for graph in graph_names:
    if lib_path != "":
        lib_path += ","
    lib_path += f"{str(build_folder)}/x86_64-linux-clang/lib{graph}.so"

backend_path = qnn_sdk_folder / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"
htp_setting_path = build_folder / "htp_setting.json"
htp_config_path = build_folder / "htp_config.json"

log_path = build_folder / args.log_file
log_file = open(log_path, "w")


def run(cmd_args: list):
    cmd = " ".join(map(str, cmd_args))
    if args.silent:
        cmd_short = " ".join(map(str, cmd_args[:1]))
        print(f"> {cmd_short}")
    else:
        print(f"> {cmd}")
    ret = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=log_file).wait()
    assert ret == 0


def generate_context_binary():
    bin_generator_path = qnn_sdk_folder / "bin" / "x86_64-linux-clang" / "qnn-context-binary-generator"
    assert bin_generator_path.exists()

    htp_setting = {
        "graphs": [{"graph_names": graph_names, "fp16_relaxed_precision": 1, "vtcm_mb": 8, "O": 3, "hvx_threads": 4}],
        "devices": [
            {
                "dsp_arch": f"v{soc_map[args.soc].htp_version}",
                "soc_model": soc_map[args.soc].soc_id,
            },
        ],
        "context": {"weight_sharing_enabled": True},
    }

    htp_config = {
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": str(htp_setting_path),
        },
        "context_configs": {"enable_graphs": graph_names},
        "graph_configs": [{"graph_name": graph_name} for graph_name in graph_names],
    }

    def export_json(obj, path: Path):
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    export_json(htp_setting, htp_setting_path)
    export_json(htp_config, htp_config_path)

    cmd_args = [
        bin_generator_path,
        "--backend",
        backend_path,
        "--model",
        lib_path,
        "--output_dir",
        build_folder,
        "--binary_file",
        args.artifact_name,
        "--config_file",
        htp_config_path,
        "--input_output_tensor_mem_type memhandle",
    ]

    run(cmd_args)


generate_context_binary()
