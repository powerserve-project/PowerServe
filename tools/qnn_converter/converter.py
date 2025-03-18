import argparse
import json
import os
import subprocess
from pathlib import Path

from soc_config import soc_map


def run_shell_command(command):
    print(f">{' '.join(command.split())}")
    ret = subprocess.Popen(command, shell=True).wait()
    assert ret == 0


def get_config_file(folder, batch_sizes):
    merged_config = {"model_parameters": {}, "qnn_parameters": {}, "graphs": [], "embeddings": []}

    for batch_size in batch_sizes:
        file_path = os.path.join(folder, f"config_batch_{batch_size}.json")
        with open(file_path, "r") as json_file:
            data = json.load(json_file)

            if not merged_config["model_parameters"]:
                merged_config["model_parameters"] = data.get("model_parameters", {})
            if not merged_config["qnn_parameters"]:
                merged_config["qnn_parameters"] = data.get("qnn_parameters", {})

            if "graphs" in data:
                merged_config["graphs"].extend(data["graphs"])
            if "embeddings" in data:
                merged_config["embeddings"].extend(data["embeddings"])
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(merged_config, f, indent=2)


def get_output_folder(folder, batch_size, htp_version):
    kv_folder = Path(folder) / "kv"
    kv_folder.mkdir(parents=True, exist_ok=True)

    qnn_sdk_folder = os.getenv("QNN_SDK_ROOT")
    run_shell_command(f"cp {qnn_sdk_folder}/lib/aarch64-android/libQnnSystem.so {folder}")
    run_shell_command(f"cp {qnn_sdk_folder}/lib/aarch64-android/libQnnHtp.so {folder}")
    run_shell_command(f"cp {qnn_sdk_folder}/lib/aarch64-android/libQnnHtpV{htp_version}Stub.so {folder}")

    htp_lib_folder = f"{qnn_sdk_folder}/lib/hexagon-v{htp_version}/unsigned"
    run_shell_command(f"cp {htp_lib_folder}/libQnnHtpV{htp_version}.so {folder}")
    run_shell_command(f"cp {htp_lib_folder}/libQnnHtpV{htp_version}Skel.so {folder}")

    if htp_version == 79:
        run_shell_command(f"cp {htp_lib_folder}/libQnnHexagonSkel_dspApp.so {folder}")

    run_shell_command(f"cp {args.build_folder}/config.json {folder}")
    run_shell_command(f"cp {args.build_folder}/m*/batch_{batch_size}/kv/* {str(kv_folder)}/")
    run_shell_command(f"cp {args.build_folder}/m*/*.bin {folder}")
    run_shell_command(f"cp {args.build_folder}/output_embedding/*.bin {folder}")


def main(args):
    for i in args.batch_sizes:
        onnx_command = f"""
        python export_to_onnx.py \
            --n-threads {args.n_threads} \
            --model-folder {args.model_folder} \
            --model-name {args.model_name} \
            --graph-name batch_{i} \
            --system-prompt-file {args.system_prompt_file} \
            --prompt-file {args.prompt_file} \
            --output-folder {args.build_folder} \
            --max-n-tokens {args.max_n_tokens} \
            --n-model-chunks {args.n_model_chunks}"""
        if args.fp16_lm_head:
            onnx_command += " --fp16-lm-head"
        run_shell_command(onnx_command)

        generate_so_command = f"""
        python build_all_layers.py \
            --build-folder {args.build_folder} \
            --batch-size {i} \
            --n-model-chunks {args.n_model_chunks} \
            --artifact-name {args.artifact_name} \
            --graph-names batch_{i} \ 
            --n-threads {args.n_threads}
        """
        run_shell_command(generate_so_command)

        # rm_command = f"rm -rf {args.build_folder}/m*/batch_{i}/data&&rm -rf {args.build_folder}/m*/batch_{i}/onnx_model"
        # run_shell_command(rm_command)

    get_config_file(args.build_folder, args.batch_sizes)
    generate_binary_command = f"""
        python build_all_layers.py \
            --build-folder {args.build_folder} \
            --artifact-name {args.artifact_name} \
            --graph-names {" ".join([f"batch_{i}" for i in args.batch_sizes])} \
            --n-model-chunks {args.n_model_chunks} \
            --soc {args.soc}
        """
    run_shell_command(generate_binary_command)

    get_output_folder(args.output_folder, args.batch_sizes[0], soc_map[args.soc].htp_version)

    # run_shell_command(f"rm -r {args.build_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the model in safetensors format to a QNN executable binary format."
    )

    parser.add_argument("--n-threads", type=int, default=24, help="Number of threads to use when exporting the model.")
    parser.add_argument("--model-folder", type=str, help="Model folder path.", required=True)
    parser.add_argument("--model-name", type=str, help="Model name.", required=True)
    parser.add_argument(
        "--system-prompt-file", type=str, default="system_prompt.txt", help="System prompt file path.", required=True
    )
    parser.add_argument("--prompt-file", type=str, default="lab_intro.md", help="Prompt file path.", required=True)
    parser.add_argument("--build-folder", type=str, default="./build")
    parser.add_argument("--output-folder", type=str, default="./output")
    parser.add_argument("--max-n-tokens", type=int, default=1000)
    parser.add_argument("--n-model-chunks", type=int, default=1, help="Number of model chunks.")
    parser.add_argument("--artifact-name", type=str, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--soc", type=str, choices=soc_map.keys(), default="8gen3")
    parser.add_argument("--fp16-lm-head", action="store_true")

    args = parser.parse_args()
    main(args)
