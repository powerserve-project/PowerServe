#!/bin/bash
# test-decode.sh /data/data/com.termux/files/home/CI u0_a342@192.168.60.173 8022 powerserve-llama3.1-8b / powerserve-qwen2-7b

DEVICE_ROOT=$1
DEVICE_URL=$2
DEVICE_PORT=$3

TARGET=$4
if [ "${TARGET}" == "" ]; then
    TARGET="powerserve-llama3.1-8b_n"
fi

USE_QNN=$5
if [ "${USE_QNN}" == "" ]; then
    USE_QNN=0
fi

STEPS=$6
if [ "${STEPS}" == "" ]; then
    STEPS=32
fi

THREADS_NUM=$7
if [ "${THREADS_NUM}" == "" ]; then
    THREADS_NUM=4
fi

PROMPT_FILE=$8
if [ "${PROMPT_FILE}" == "" ]; then
    PROMPT_FILE="hello.txt"
fi

WORK_FOLDER="${DEVICE_ROOT}/${TARGET}"

function help() {
    echo "Usage: $0 <device_root> <device_url> <device_port> [-] [target] [use_qnn]"
    exit 1
}

function clean() {
    echo "finish"
}

if [ $# -lt 3 ]; then
    help
fi

set -e
trap clean EXIT

ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    echo '>>>>>>>>>>>> Run test. <<<<<<<<<<<<';
"

set -x
if [ "${USE_QNN}" == "1" ]; then
    ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
        export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 && ${WORK_FOLDER}/bin/powerserve-run -d ${WORK_FOLDER};
    "
else
    ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
        ${WORK_FOLDER}/bin/powerserve-run -d ${WORK_FOLDER} --no-qnn -n 32;
    "
fi
set +x

ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    echo '>>>>>>>>>>>> Run over. <<<<<<<<<<<<';
"
