#!/bin/bash
# test-speculative.sh /data/data/com.termux/files/home/CI u0_a342@192.168.60.173 8022 powerserve-llama3.1-8b-spec

DEVICE_ROOT=$1
DEVICE_URL=$2
DEVICE_PORT=$3

TARGET=$4
if [ "${TARGET}" == "" ]; then
    TARGET="powerserve-llama3.1-8b-spec"
fi

USE_QNN=$5
if [ "${USE_QNN}" == "" ]; then
    USE_QNN=1
fi

STEPS=$6
if [ "${STEPS}" == "" ]; then
    STEPS=500
fi

WORK_FOLDER="${DEVICE_ROOT}/${TARGET}"
PROMPT_FILE="math.txt"

function help() {
    echo "Usage: \$0 <device_root> <device_url> <device_port> [-] [target] [use_qnn] [steps]"
    exit 1
}

function clean() {
    echo "pass"
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
        export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 && ${WORK_FOLDER}/bin/powerserve-run -d ${WORK_FOLDER} -n 64 -p \"Please tell a story:\" --use-spec;
    "
else
    echo "No support"
fi
set +x

ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    echo '>>>>>>>>>>>> Run over. <<<<<<<<<<<<';
"
