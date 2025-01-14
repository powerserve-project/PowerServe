# Get the code

`git clone https://github.com/powerserve-project/PowerServe`

# Enter the directory

`cd PowerServe`

Please make sure you locates at the PowerServe directory.

# Run the end-to-end Powerserve script.

Note that we have a few prerequisites for the end-to-end script:
- Your computer should have docker installed. [How to install docker](https://docs.docker.com/get-started/get-docker/)
- Your phone should open ADB debugging and connect to the computer. [How to enable ADB debugging](https://developer.android.google.cn/tools/adb) When you typing `adb shell` in your shell, you should see the shell of your phone.
- Your Internet connection with github, docker and huggingface should be good. If you need to use a proxy, you can set the `https_proxy` at your host shell environment. If `https_proxy` is set, the end-to-end script will use the proxy to download the model and docker images automatically.

These prerequisites are necessary for the end-to-end script to run successfully, no matter what operating system you are using.

## Linux or MacOS or WSL(Windows Subsystem for Linux)

Table of supported models:
| Model Name | Huggingface Link | Speculation Support |
| ---------- | ----------- | ------------------- |
| smallthinker-3b | [SmallThinker-3B](https://huggingface.co/PowerServe/SmallThinker-3B-PowerServe-QNN29-8G4) | Yes |
| llama-3.1-8b | [Llama-3.1-8B](https://huggingface.co/PowerServe/Llama-3.1-8B-PowerServe-QNN29-8G4) | Yes |
| llama-3.2-1b | [Llama-3.2-1B](https://huggingface.co/PowerServe/Llama-3.2-1B-PowerServe-QNN29-8G4) | No |
| qwen-2.5-3b | [Qwen-2.5-3B](https://huggingface.co/PowerServe/Qwen-2.5-3B-PowerServe-QNN29-8G4) | No |
| qwen-2-0.5b | [Qwen-2-0.5B](https://huggingface.co/PowerServe/Qwen-2-0.5B-PowerServe-QNN29-8G4) | No |


```
PowerServe End-to-End Script

positional arguments:
  {compile,run,clean}
    compile            Compile the binary using Docker
    run                Run the model on the phone
                       Supported models: smallthinker-3b, llama-3.1-8b, llama-3.2-1b, qwen-2.5-3b, qwen-2-0.5b
                       Models supporting speculation: smallthinker-3b, llama-3.1-8b
    clean              Clean all environment(local and phone)

options:
  -h, --help           show this help message and exit
```

The above is the help message of the end-to-end script. You can run the script with the following steps:

1. `./tools/end_to_end/powerserve.sh compile`. This step compile the binary file for the phone using docker. If you run this script for the first time, it will take a while to download the docker image.
2. `./tools/end_to_end/powerserve.sh run llama-3.2-1b`. This step will create the workspace and push the files to the phone. It may take several to minutes to download the model if you run this script for the first time.

Explanation of all parameters of the run script:

```
usage: powerserve.sh run [-h] [-p PROMPT] [-f PROMPT_FILE] [-s] [-c] model_name

positional arguments:
  model_name            Name of the model to run

options:
  -h, --help            show this help message and exit
  -p PROMPT, --prompt PROMPT
                        Prompt text
  -f PROMPT_FILE, --prompt-file PROMPT_FILE
                        File to read prompt from
  -s, --speculation     Enable speculation
  -c, --cpu-only        Use CPU only
```

## Windows

If your host is Windows, we strongly recommend you to use WSL(Windows Subsystem for Linux) to run the end-to-end script. Then you can follow the Linux or MacOS or WSL instructions to run the script.

If your adb in WSL cannot detect your phone, you can go to this link to see if it is helpful: [ADB device list empty using WSL2](https://stackoverflow.com/questions/60166965/adb-device-list-empty-using-wsl2)
