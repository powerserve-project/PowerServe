#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from subprocess import DEVNULL

import requests
from huggingface_hub import snapshot_download


# fmt: off
# Define model mappings and speculation support
MODEL_MAP = {
    "smallthinker-3b": "PowerServe/SmallThinker-3B-PowerServe-QNN29-{soc_name}",
    "llama-3.1-8b": "PowerServe/Llama-3.1-8B-PowerServe-QNN29-{soc_name}",
    "llama-3.2-1b": "PowerServe/Llama-3.2-1B-PowerServe-QNN29-{soc_name}",
    "qwen-2.5-3b": "PowerServe/Qwen-2.5-3B-PowerServe-QNN29-{soc_name}",
    "qwen-2-0.5b": "PowerServe/Qwen-2-0.5B-PowerServe-QNN29-{soc_name}",
    "internlm-3-8b": "PowerServe/InternLM-3-8B-PowerServe-QNN29-{soc_name}",
    "deepseek-r1-llama-8b": "PowerServe/DeepSeek-R1-Distill-Llama-8B-PowerServe-QNN29-{soc_name}",
}

# Updated SPECULATION_MAP to directly store model_name to {target_model, draft_model} mapping
SPECULATION_MAP = {
    "smallthinker-3b": {
        "target_model": "PowerServe/SmallThinker-3B-PowerServe-QNN29-{soc_name}",
        "draft_model": "PowerServe/SmallThinker-0.5B-PowerServe-QNN29-{soc_name}",
    },
    "llama-3.1-8b": {
        "target_model": "PowerServe/Llama-3.1-8B-PowerServe-Speculate-QNN29-{soc_name}",
        "draft_model": "PowerServe/Llama-3.2-1B-PowerServe-QNN29-{soc_name}",
    },
    # "deepseek-r1-llama-8b": {
    #     "target_model": "PowerServe/DeepSeek-R1-Distill-Llama-8B-PowerServe-QNN29-{soc_name}",
    #     "draft_model": "PowerServe/DeepSeek-R1-Distill-Llama-8B-PowerServe-Speculate-QNN29-{soc_name}",
    # },
}

SUPPORTED_MODELS = list(MODEL_MAP.keys())
MODELS_WITH_SPECULATION = list(SPECULATION_MAP.keys())


def check_network_connectivity(url):
    try:
        response = requests.head(url, timeout=10)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False


def get_soc():
    adb_devices = subprocess.check_output(["adb", "devices"]).decode().splitlines()
    if len(adb_devices) <= 2:
        print("\033[31mNo devices found in ADB\033[0m")
        print("\033[31mCheck whether the Phone is connected with suitable wire.\033[0m")
        print(
            "\033[31mIf you don't know how to apply ADB on your phone. You can go to link https://developer.android.google.cn/tools/adb \033[0m"
        )
        sys.exit(1)
    if len(adb_devices) > 3:
        print("\033[31mToo many devices found in ADB\031m")
        print("\033[31mPlease disconnect the unnecessary devices.\033[0m")
        sys.exit(1)
    platform_name_list = ["pineapple", "sun"]
    platform_name_8G3_list = ["pineapple"]
    platform_name_8G4_list = ["sun"]

    platform_name = subprocess.check_output(["adb", "shell", "getprop", "ro.board.platform"]).decode().strip()
    if platform_name not in platform_name_list:
        print(f"\033[31mPlatform name {platform_name} is not supported.\033[0m")
        print(f"\033[31mSupported platform names: {platform_name_list} (which means 8G3 or 8G4)\033[0m")
        sys.exit(1)

    soc_name = "8G3" if platform_name in platform_name_8G3_list else "8G4"
    print(f"\033[36mDetermining SoC\033[0m \033[32m[OK]\033[0m")
    print(f"\033[32mSoC: {soc_name}\033[0m")
    return soc_name


def compile_binary():
    if not check_network_connectivity("https://github.com"):
        print("\033[31mGitHub is not reachable. Please check your internet connection.\033[0m")
        sys.exit(1)

    print("\033[36mDownloading submodules from GitHub\033[0m")
    subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=True)
    print("\033[36mDownloading submodules\033[0m \033[32m[OK]\033[0m")

    docker_command = [
        "sudo",
        "docker",
        "run",
        "--platform",
        "linux/amd64",
        "--rm",
        "--name",
        "powerserve_container",
        "-v",
        f"{os.getcwd()}:/code",
        "-w",
        "/code",
        "-e",
        f'https_proxy={os.environ.get("https_proxy", "")}',
        "-e",
        f'http_proxy={os.environ.get("http_proxy", "")}',
        "-e",
        f'socks_proxy={os.environ.get("socks_proxy", "")}',
        "--network",
        "host",
        "santoxin/mobile-build:v1.1",
        "/bin/bash",
        "-c",
        "./tools/end_to_end/compile.sh",
    ]

    print("\033[36mCompiling the binary using Docker\033[0m")
    subprocess.run(docker_command, check=True)
    print("\033[36mCompiling the binary\033[0m \033[32m[OK]\033[0m")


def check_dir_on_phone(dir_path):
    try:
        result = subprocess.run(["adb", "shell", f"ls {dir_path}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return True
    except subprocess.CalledProcessError:
        pass
    return False


def run_model(args):
    # print("\033[36mStarting to run the model on the phone\033[0m")

    if not os.path.exists("./build_android"):
        print(
            "\033[31m./build_android directory not found. Did you forget to run `./tools/end_to_end/powerserve.sh compile`?\033[0m"
        )
        sys.exit(1)

    soc_name = get_soc()

    if args.model_name not in SUPPORTED_MODELS:
        print(f"\033[31mModel {args.model_name} is not supported.\033[0m")
        print(f"\033[31mSupported models: {', '.join(SUPPORTED_MODELS)}\033[0m")
        sys.exit(1)

    if args.speculation:
        if args.model_name in SPECULATION_MAP:
            model_info = SPECULATION_MAP[args.model_name]
            target_model = model_info["target_model"].format(soc_name=soc_name)
            draft_model = model_info["draft_model"].format(soc_name=soc_name)
            models = [target_model, draft_model]
        else:
            print(f"\033[31mSpeculation not supported for model {args.model_name}.\033[0m")
            print(f"\033[31mModels supporting speculation: {', '.join(MODELS_WITH_SPECULATION)}.\033[0m")
            sys.exit(1)
    else:
        target_model = MODEL_MAP[args.model_name].format(soc_name=soc_name)
        models = [target_model]

    if not check_network_connectivity("https://huggingface.co"):
        print("\033[31mHugging Face is not reachable. Please check your internet connection.\033[0m")
        sys.exit(1)

    cache_dir = os.path.join(os.getcwd(), ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    for model_repo in models:
        model_dir = os.path.join("models", model_repo.split("/")[-1])
        print(f"\033[36mDownloading model {model_repo} to {model_dir}\033[0m")
        snapshot_download(repo_id=model_repo, local_dir=model_dir, local_dir_use_symlinks=False, cache_dir=cache_dir)
        print(f"\033[36mDownloading model {model_repo}\033[0m \033[32m[OK]\033[0m")
    if args.prompt_file:
        try:
            with open(args.prompt_file, "r") as f:
                prompt = f.read().strip()
            if not prompt:
                print("Prompt is empty. Please provide a valid prompt.")
                sys.exit(1)
        except FileNotFoundError:
            print(f"\033[31mFile {args.prompt_file} not found.\033[0m")
            sys.exit(1)
        except IOError:
            print(f"\033[31mError reading file {args.prompt_file}.\033[0m")
            sys.exit(1)
    else:
        prompt = args.prompt

    args.prompt = prompt

    print("\033[32mModel: ", args.model_name, "\033[0m")
    print("\033[32mPrompt: ", args.prompt, "\033[0m")
    if args.cpu_only:
        print("\033[32mCPU only: ", args.cpu_only, "\033[0m")
    if args.speculation:
        print("\033[32mSpeculation: ", args.speculation, "\033[0m")
    if "smallthinker" in args.model_name or "qwen" in args.model_name:
        args.prompt = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
    elif "llama" in args.model_name:
        args.prompt = f"<|start_header_id|>user<|end_header_id|>\n{args.prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
    elif "internlm" in args.model_name:
        thinking_system_prompt="<|im_start|>system\nYou are an expert mathematician with extensive experience in mathematical competitions. You approach problems through systematic thinking and rigorous reasoning. When solving problems, follow these thought processes:\n## Deep Understanding\nTake time to fully comprehend the problem before attempting a solution. Consider:\n- What is the real question being asked?\n- What are the given conditions and what do they tell us?\n- Are there any special restrictions or assumptions?\n- Which information is crucial and which is supplementary?\n## Multi-angle Analysis\nBefore solving, conduct thorough analysis:\n- What mathematical concepts and properties are involved?\n- Can you recall similar classic problems or solution methods?\n- Would diagrams or tables help visualize the problem?\n- Are there special cases that need separate consideration?\n## Systematic Thinking\nPlan your solution path:\n- Propose multiple possible approaches\n- Analyze the feasibility and merits of each method\n- Choose the most appropriate method and explain why\n- Break complex problems into smaller, manageable steps\n## Rigorous Proof\nDuring the solution process:\n- Provide solid justification for each step\n- Include detailed proofs for key conclusions\n- Pay attention to logical connections\n- Be vigilant about potential oversights\n## Repeated Verification\nAfter completing your solution:\n- Verify your results satisfy all conditions\n- Check for overlooked special cases\n- Consider if the solution can be optimized or simplified\n- Review your reasoning process\nRemember:\n1. Take time to think thoroughly rather than rushing to an answer\n2. Rigorously prove each key conclusion\n3. Keep an open mind and try different approaches\n4. Summarize valuable problem-solving methods\n5. Maintain healthy skepticism and verify multiple times\nYour response should reflect deep mathematical understanding and precise logical thinking, making your solution path and reasoning clear to others.\nWhen you're ready, present your complete solution with:\n- Clear problem understanding\n- Detailed solution process\n- Key insights\n- Thorough verification\nFocus on clear, logical progression of ideas and thorough explanation of your mathematical reasoning. Provide answers in the same language as the user asking the question, repeat the final answer using a '\\boxed{}' without any units, you have [[3800]] tokens to complete the answer.\n<|im_end|>\n"
        args.prompt = f"{thinking_system_prompt}<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"

    working_dir = "."
    if args.speculation:
        print("\033[32mTarget model: ", models[0].split("/")[-1], "\033[0m")
        print("\033[32mDraft model: ", models[1].split("/")[-1], "\033[0m")
        create_command = f'python3 {working_dir}/powerserve create --no-extract-qnn -m "{working_dir}/models/{models[0].split("/")[-1]}" -d "{working_dir}/models/{models[1].split("/")[-1]}" --exe-path {working_dir}/build_android/out'
    else:
        create_command = f'python3 {working_dir}/powerserve create --no-extract-qnn -m "{working_dir}/models/{target_model.split("/")[-1]}" --exe-path {working_dir}/build_android/out'

    print("\033[36mCreating the workspace\033[0m")
    host_command = [
        "python3",
        "powerserve",
        "create",
        "--no-extract-qnn",
        "-m",
        f'models/{target_model.split("/")[-1]}',
    ]
    if args.speculation:
        host_command.extend(["-d", f'models/{models[1].split("/")[-1]}'])
    host_command.extend(["--exe-path", "build_android/out"])

    subprocess.run(host_command, check=True)
    print("\033[36mCreating the workspace\033[0m \033[32m[OK]\033[0m")

    deploy_to_phone(args, models)
    print("\033[36mSuccessfully finished running the model on the phone\033[0m \033[32m[OK]\033[0m")


def deploy_to_phone(args, models):
    target_path = "/data/local/tmp"
    # mkdir proj/
    if not check_dir_on_phone(f"{target_path}/proj"):
        subprocess.run(["adb", "shell", f"mkdir -p {target_path}/proj"], check=True)

    for model_repo in models:
        model_name = model_repo.split("/")[-1]
        phone_model_path = f"{target_path}/proj/{model_name}"
        if not check_dir_on_phone(phone_model_path):
            print(f"\033[36mPushing model {model_name} to the phone\033[0m")
            subprocess.run(["adb", "push", "--sync", f"./models/{model_name}", f"{target_path}/proj/"], check=True)
            print(f"\033[36mPushing model {model_name}\033[0m \033[32m[OK]\033[0m")
        else:
            print(f"\033[36mModel {model_name} already exists on the phone. Skipping model push.\033[0m")

    subprocess.run(
        ["adb", "push", "--sync", "./proj/qnn_libs", f"{target_path}/proj/"], check=True, stdout=DEVNULL, stderr=DEVNULL
    )
    subprocess.run(["adb", "push", "--sync", "./proj/hparams.json", f"{target_path}/proj/"], check=True)
    subprocess.run(["adb", "push", "--sync", "./proj/workspace.json", f"{target_path}/proj/"], check=True)
    subprocess.run(["adb", "push", "--sync", "./proj/bin", f"{target_path}/proj/"], check=True)
    subprocess.run(["adb", "shell", f"chmod +x {target_path}/proj/bin/*"], check=True)

    command = f'adb shell "{target_path}/proj/bin/powerserve-run -d {target_path}/proj -n {args.n_predicts} -p \\"{args.prompt}\\"'
    if args.speculation:
        command += " --use-spec"
    if args.cpu_only:
        command += " --no-qnn"
    command += '"'

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("\033[31mError occurred while running the model on the phone.\033[0m")
        print("\033[31mThe process might have aborted. Please check the phone's logcat for more details.\033[0m")
        print("\033[31mExiting gracefully...\033[0m")
        sys.exit(0)


def clean_environment():
    local_dirs = ["./proj", "./build_android"]
    phone_dir = "/data/local/tmp/proj"

    print("\033[36mCleaning the environment\033[0m")

    for dir_path in local_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"\033[36mCleaning local directory {dir_path}\033[0m \033[32m[OK]\033[0m")
        else:
            print(f"\033[36mLocal directory {dir_path} does not exist. Skipping.\033[0m")

    if check_dir_on_phone(phone_dir):
        subprocess.run(["adb", "shell", f"rm -rf {phone_dir}"], check=True)
        print(f"\033[36mCleaning phone directory {phone_dir}\033[0m \033[32m[OK]\033[0m")
    else:
        print(f"\033[36mPhone directory {phone_dir} does not exist. Skipping.\033[0m")


def main():
    if not os.path.exists("tools"):
        print("\033[31mPlease run this script from the root directory of the PowerServe project.\033[0m")
        sys.exit(1)

    model_help = f"Supported models: {', '.join(SUPPORTED_MODELS)}\n"
    model_help += f"Models supporting speculation: {', '.join(MODELS_WITH_SPECULATION)}"

    parser = argparse.ArgumentParser(
        description="PowerServe End-to-End Script", formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command")

    compile_parser = subparsers.add_parser("compile", help="Compile the binary using Docker")

    smallthinker_prompt = "Which one is larger? 9.11 or 9.8"
    default_prompt = "What is the capital of France?"
    run_parser = subparsers.add_parser("run", help=f"Run the model on the phone\n{model_help}")
    run_parser.add_argument("-p", "--prompt", default=default_prompt, help="Prompt to use")
    run_parser.add_argument("-f", "--prompt-file", help="File to read prompt from")
    run_parser.add_argument("-n", "--n-predicts", type=int, default=1500, help="Number of predictions to make")
    run_parser.add_argument("-s", "--speculation", action="store_true", default=False, help="Enable speculation")
    run_parser.add_argument("-c", "--cpu-only", action="store_true", default=False, help="Use CPU only")
    run_parser.add_argument("model_name", help="Name of the model to run")

    clean_parser = subparsers.add_parser("clean", help="Clean all environment(local and phone)")

    args = parser.parse_args()
    
    # if -s && -c then print error
    if args.speculation and args.cpu_only:
        print("\033[31mSpeculation only works with NPU, please remove -c/--cpu-only flag.\033[0m")
        sys.exit(1)

    try:
        if args.command == "compile":
            compile_binary()
            print("\033[36mCompilation complete\033[0m \033[32m[OK]\033[0m")
        elif args.command == "run":
            run_model(args)
        elif args.command == "clean":
            clean_environment()
            print("\033[36mCleanup complete\033[0m \033[32m[OK]\033[0m")
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\033[33mKeyboardInterrupt detected. Exiting gracefully...\033[0m")
        sys.exit(0)


if __name__ == "__main__":
    main()
# fmt: on
