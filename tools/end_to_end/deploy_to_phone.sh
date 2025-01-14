#!/bin/bash

prompt="In recent years, the landscape of artificial intelligence has been significantly transformed by the advent of large language models (LLMs). Traditionally, these models have been deployed in cloud environments due to their computational demands. However, the emergence of on-edge LLMs is reshaping how AI can be utilized at the edge of networks, offering numerous advantages in terms of latency, privacy, and accessibility."
speculation_enabled="no"
cpu_only="no"
soc_name=""

# read soc_name from tmpfile
soc_name=$(cat tmpfile)

while getopts ":p:sc" opt; do
    case ${opt} in
        p )
            prompt=$OPTARG
            ;;
        s )
            speculation_enabled="yes"
            ;;
        c )
            cpu_only="yes"
            ;;
        \? )
            echo -e "\033[31mInvalid option: -$OPTARG\033[0m" 1>&2
            exit 1
            ;;
        : )
            echo -e "\033[31mInvalid option: -$OPTARG requires an argument\033[0m" 1>&2
            exit 1
            ;;
    esac
done

# Deploy the ./proj folder to the phone
echo -e "\033[32mDeploying to phone\033[0m"

TARGET_PATH="/data/local/tmp"

adb push ./proj $TARGET_PATH/

adb shell "chmod +x $TARGET_PATH/proj/bin/*"

# check if adb push is successful
if [ $? -ne 0 ]; then
    echo -e "\033[31mFailed to push the project to the phone. Check ADB.\033[0m"
    exit 1
fi

if [ "$speculation_enabled" == "yes" ]; then
    adb shell "$TARGET_PATH/proj/bin/powerserve-run -d $TARGET_PATH/proj -n 500 -p \"$prompt\" --use-spec"
else
    adb shell "$TARGET_PATH/proj/bin/powerserve-run -d $TARGET_PATH/proj -n 500 -p \"$prompt\""
fi
