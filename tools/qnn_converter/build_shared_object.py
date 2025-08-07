import argparse
import json
import os
import subprocess
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--encoding", type=Path, required=True)
parser.add_argument("--io-spec", type=Path, required=True)
parser.add_argument("--input-list", type=Path, required=True)
parser.add_argument("--output-folder", type=Path, required=True)
parser.add_argument("--artifact-name", type=str, required=True)
parser.add_argument("--log-file", type=str, default="build_so.log")
parser.add_argument("--graph-names", type=str, nargs="+", required=True)
parser.add_argument("--generate-binary", type=bool)
parser.add_argument("--silent", action="store_true", help="Hide the shell command arguments.")
args = parser.parse_args()

output_folder: Path = args.output_folder
output_folder.mkdir(parents=True, exist_ok=True)

qnn_sdk_folder = os.getenv("QNN_SDK_ROOT")
assert qnn_sdk_folder is not None, "QNN_SDK_ROOT is not set"
qnn_sdk_folder = Path(qnn_sdk_folder)

model_path: Path = args.model
encoding_path: Path = args.encoding
io_spec_path: Path = args.io_spec
input_list_path: Path = args.input_list
cpp_path = model_path.with_suffix(".cpp")
bin_path = model_path.with_suffix(".bin")
graph_name = model_path.stem

log_path = output_folder / args.log_file
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


def convert_model():
    converter_path = qnn_sdk_folder / "bin" / "x86_64-linux-clang" / "qnn-onnx-converter"
    assert converter_path.exists()

    cmd_args = [
        converter_path,
        "--input_network",
        model_path,
        "--quantization_overrides",
        encoding_path,
        "--output_path",
        cpp_path,
        "--preserve_io layout",
        "--bias_bitwidth 32",
        "--act_bitwidth 16",
        "--weights_bitwidth 4",
        # '--no_simplification',
        "--use_per_channel_quantization",
        "--use_per_row_quantization",
        # "--keep_weights_quantized",
        "--param_quantizer_schema symmetric",
        "--act_quantizer_schema asymmetric",
        "--input_list",
        input_list_path,
        "--debug"
    ]

    with open(io_spec_path, "r") as f:
        io_spec = json.load(f)
        assert isinstance(io_spec, list)

    for info in io_spec:
        name = info["name"]

        if info["type"] == "in":
            cmd_args += [
                "--input_dtype",
                "'" + name + "'",
                info["dtype"],
                "--input_dim",
                "'" + name + "'",
                ",".join(map(str, info["shape"])),
            ]
        elif info["type"] == "out":
            cmd_args += ["--out_name", "'" + name + "'"]

        if info["preserve_dtype"]:
            cmd_args += ["--preserve_io datatype", "'" + name + "'"]

    run(cmd_args)


def generate_library():
    lib_generator_path = qnn_sdk_folder / "bin" / "x86_64-linux-clang" / "qnn-model-lib-generator"
    assert lib_generator_path.exists()

    cmd_args = [
        lib_generator_path,
        "--cpp",
        cpp_path,
        "--bin",
        bin_path,
        "--lib_targets x86_64-linux-clang",
        "--lib_name",
        graph_name,
        "--output_dir",
        output_folder,
        "--debug"
    ]

    run(cmd_args)


convert_model()
generate_library()
