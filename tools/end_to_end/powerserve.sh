#!/bin/bash

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "\033[31m$1 could not be found. Please install it.\033[0m"
        exit 1
    fi
}

# Check necessary commands
check_command "docker"
check_command "adb"
check_command "python3"
check_command "pip"

# echo -e "\033[36mChecking if the needed PowerServe docker image was updated...\033[0m"
# don not show stderr or stdout
sudo docker pull santoxin/mobile-build:v1.1 &> /dev/null

# Check whether now locates at .../PowerServe
if [ ! -d "tools" ]; then
    echo -e "\033[31mPlease run this script from the root directory of PowerServe.\033[0m"
    exit 1
fi

# check if pip install requests huggingface_hub
pip install requests huggingface_hub --quiet

# forward all strings to python file
# example: ./tools/powerserve.sh run llama-3.2-1b -> then call python ./tools/end_to_end/powerserve.py run llama-3.2-1b
python3 ./tools/end_to_end/powerserve.py "$@"
