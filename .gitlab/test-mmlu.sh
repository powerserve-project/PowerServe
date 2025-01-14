#!/bin/bash
# test-mmlu.sh /data/data/com.termux/files/home/CI u0_a342 192.168.60.173 8022 <container-name>

DEVICE_ROOT=$1
DEVICE_USER=$2
DEVICE_HOST=$3
DEVICE_PORT=$4
CONTAINER_NAME=$5

TARGET=$6
if [ "${TARGET}" == "" ]; then
    TARGET="powerserve-llama3.1-8b"
fi

SERVER_HOST=${DEVICE_HOST}
SERVER_PORT="18080"
DEVICE_URL="${DEVICE_USER}@${DEVICE_HOST}"

WORK_FOLDER="${DEVICE_ROOT}/${TARGET}"

function help() {
    echo "Usage: $0 <device_root> <device_host> <device_user> <device_port> <mmlu_client_container_name> - [target]"
    exit 1
}

function clean() {
    set +x
    source ./.gitlab/common.sh
    # temp_disable_errexit try_twice 10 ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    #     echo '>>>>>>>>>>>> Stop server. <<<<<<<<<<<<';
    #     sudo ps -e -o comm= | grep 'powerserve-' |xargs -n 1 echo;
    #     sudo pkill powerserve-;
    #     sleep 3;
    #     echo '>>>>>>>>>>>> Stop server over. <<<<<<<<<<<<';
    #     sudo ps -e -o comm= | grep 'powerserve-' |xargs -n 1 echo
    # "
}

if [ $# -lt 5 ]; then
    help
fi

set -e
trap clean EXIT
set -x

echo '>>>>>>>>>>>> Start server. <<<<<<<<<<<<';
echo "No Server Test"
echo '>>>>>>>>>>>> Start server over. <<<<<<<<<<<<';

sleep 10

echo '>>>>>>>>>>>> Test mmlu. <<<<<<<<<<<<';
# sudo podman exec -it ${CONTAINER_NAME} bash -c -i "
#     cd /code/tools/mmlu;
#     python ./mmlu_test.py --host ${SERVER_HOST} --port ${SERVER_PORT} -s 1
# "
echo '>>>>>>>>>>>> Test mmlu over. <<<<<<<<<<<<';

set +x
