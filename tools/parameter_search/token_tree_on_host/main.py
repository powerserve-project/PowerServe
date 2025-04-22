# Copyright 2024-2025 PowerServe Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import subprocess
from pathlib import Path
from itertools import product
import tqdm


# Usage: python main.py
# The search_config should be modified according to the user's needs
# The prompt files should be placed in the correct path in advance


def dict_product(input_dict):
    keys = input_dict.keys()
    values = input_dict.values()
    return [dict(zip(keys, combination)) for combination in product(*values)]


search_config = {
    "device_folder": "/data/data/com.ps",
    "target_model_name": "llama3_1_8b",
    "target_batch_size": 12,
    "draft_model_name": "llama3_2_1b",
    "prompt_file": "gsm8k_llama.txt",
    "search_space": {
        "draft-sampler-top-k": [10, 15, 20],
        "draft-sampler-temperature": [1.2, 1.4, 1.5, 1.6, 1.7, 1.8],
        "draft-sampler-p-base": [0.6, 0.75, 0.9],
        "token-tree-max-fan-out": [3, 5, 6, 7, 8, 9],
        "token-tree-min-prob": [0.1, 0.15, 0.175, 0.2, 0.25],
        "token-tree-early-stop": [1],
    },
}


profile_result = {}
search_result = []


def run_shell_command(command):
    print(f"> {' '.join(command.split())}")
    ret = subprocess.Popen(command, shell=True).wait()
    assert ret == 0

for model in ["target", "draft"]:
    workspace = {"executables": "bin", "hparams_config": "hparams.json", "model_main": f"""{search_config["target_model_name"] if model is "target" else search_config["draft_model_name"]}""", "model_draft": ""}
    with open("./dummy_workspace.json", 'w') as file:
        json.dump(workspace, file, indent=4)
    cmd = f"adb push ./dummy_workspace.json {str(Path(search_config['device_folder'])/"workspace.json")}"
    run_shell_command(cmd)

    cmd = f"adb shell touch {str(Path(search_config['device_folder'])/'search_log.txt')}"
    run_shell_command(cmd)

    cmd = f"""adb shell \
        \"export dump_file={str(Path(search_config['device_folder'])/'search_log.txt')} && \
        export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 &&\
        cd {str(Path(search_config['device_folder']))} && {str(Path(search_config['device_folder'])/'bin'/'run')} \
        -d {str(Path(search_config['device_folder']))} \
        --prompt-file {str(Path(search_config['device_folder'])/'prompts'/f'{search_config["prompt_file"] if model is "draft" else "long.txt"}')} \
        -n {256 if model is "draft" else 4}\" """
    run_shell_command(cmd)

    cmd = f"adb pull {str(Path(search_config['device_folder'])/'search_log.txt')} ."
    run_shell_command(cmd)

    with open('./search_log.txt', 'r') as file:
        data = json.load(file)
    data["time_per_batch"] = data["prefill_time"] / data["prefill_tokens"] * search_config["target_batch_size"] if model is "target" else data["decode_time"] / data["decode_tokens"]
    profile_result[model] = data

    cmd = f"adb shell rm -rf {str(Path(search_config['device_folder'])/'search_log.txt')}"
    run_shell_command(cmd)
    cmd = "rm -rf ./search_log.txt"
    run_shell_command(cmd)

target_base_time = profile_result["target"]["time_per_batch"]
draft_base_time = profile_result["draft"]["time_per_batch"]

search_jobs = dict_product(search_config["search_space"])
workspace = {"executables": "bin", "hparams_config": "hparams.json", "model_main": f'{search_config["target_model_name"]}', "model_draft": f'{search_config["draft_model_name"]}'}
with open("./dummy_workspace.json", 'w') as file:
    json.dump(workspace, file, indent=4)
cmd = f"adb push ./dummy_workspace.json {str(Path(search_config['device_folder'])/"workspace.json")}"
run_shell_command(cmd)
cmd = "rm -rf ./dummy_workspace.json"
run_shell_command(cmd)

for search_job in tqdm(search_jobs):
    cmd = f"adb shell touch {str(Path(search_config['device_folder'])/'search_log.txt')}"
    run_shell_command(cmd)

    cmd = f"""adb shell \
        \"export dump_file={str(Path(search_config['device_folder'])/'search_log.txt')} && \
        export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 &&\
        cd {str(Path(search_config['device_folder']))} && {str(Path(search_config['device_folder'])/'bin'/'run')} \
        -d {str(Path(search_config['device_folder']))} \
        --prompt-file {str(Path(search_config['device_folder'])/'prompts'/f'{search_config["prompt_file"]}')} \
        -n 256 \
        --use-spec"""
    for key, value in search_job.items():
        cmd += (f" --{key} {value}")
    cmd += '\"'
    run_shell_command(cmd)

    cmd = f"adb pull {str(Path(search_config['device_folder'])/'search_log.txt')} ."
    run_shell_command(cmd)

    with open('./search_log.txt', 'r') as file:
        data = json.load(file)
    estimated_speed = data["n_generated_tokens"] / ((data["n_iterations"] + data["n_draft_times"]) * draft_base_time + data["n_iterations"] * target_base_time) * 1000
    search_result.append({"search_params": search_job, "result": data, "estimated_speed": estimated_speed})

    cmd = f"adb shell rm -rf {str(Path(search_config['device_folder'])/'search_log.txt')}"
    run_shell_command(cmd)
    cmd = "rm -rf ./search_log.txt"
    run_shell_command(cmd)

search_result.sort(key=lambda x: x["estimated_speed"], reverse=True)

search_dict = {"base_speed": profile_result, "search_results": search_result}
with open("result.txt", "w") as file:
    for key, value in search_dict.items():
        file.write(f"{key}={value}\n")
