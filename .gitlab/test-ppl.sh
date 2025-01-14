#!/bin/bash

DEVICE_ROOT=$1
DEVICE_URL=$2
DEVICE_PORT=$3

TARGET=$4
if [ "${TARGET}" == "" ]; then
    TARGET="powerserve-llama3.1-8b"
fi

WORK_FOLDER="${DEVICE_ROOT}/${TARGET}"
PROMPT_FILE="${DEVICE_ROOT}/prompts/wikitext-2-small.csv"


function help() {
    echo "Usage: $0 <device_root> <device_url> <device_port> - [target]"
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
source .gitlab/common.sh


echo '>>>>>>>>>>>> Test ppl. <<<<<<<<<<<<';
set -x
ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    ${WORK_FOLDER}/bin/powerserve-perplexity-test -d ${WORK_FOLDER} -p ${PROMPT_FILE};
"
set +x
echo '>>>>>>>>>>>> Test ppl over. <<<<<<<<<<<<';
