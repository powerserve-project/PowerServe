DEVICE_ROOT=$1
DEVICE_USER=$2
DEVICE_HOST=$3
DEVICE_PORT=$4
CONTAINER_NAME=$5

TARGET=$6
if [ "${TARGET}" == "" ]; then
    TARGET="powerserve-internvl2-1b"
fi

SERVER_HOST=${DEVICE_HOST}
SERVER_PORT="28081"
DEVICE_URL="${DEVICE_USER}@${DEVICE_HOST}"

WORK_FOLDER="${DEVICE_ROOT}/${TARGET}"

function help() {
    echo "Usage: $0 <device_root> <device_host> <device_user> <device_port> <mmmu_client_container_name> - [target]"
    exit 1
}

function clean() {
    set +x
    source ./.gitlab/common.sh
    temp_disable_errexit try_twice 10 ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
        echo '>>>>>>>>>>>> Stop server. <<<<<<<<<<<<';
        sudo ps -e -o comm= | grep 'powerserve-' |xargs -n 1 echo;
        sudo pkill powerserve-;
        sleep 3;
        echo '>>>>>>>>>>>> Stop server over. <<<<<<<<<<<<';
        sudo ps -e -o comm= | grep 'powerserve-' |xargs -n 1 echo
    "
}

if [ $# -lt 5 ]; then
    help
fi

set -e
trap clean EXIT
set -x

echo '>>>>>>>>>>>> Start server. <<<<<<<<<<<<';
ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    ${DEVICE_ROOT}/powerserve server \
    --host ${SERVER_HOST} \
    --port ${SERVER_PORT} \
    -d ${WORK_FOLDER} \
    >/dev/null 2>&1
" &
echo '>>>>>>>>>>>> Start server over. <<<<<<<<<<<<';

sleep 10

echo '>>>>>>>>>>>> Test mmmu. <<<<<<<<<<<<';
sudo podman exec -it ${CONTAINER_NAME} bash -c -i "
    cd /code/tools/mmmu_test;
    python ./mmmu_test.py --host ${SERVER_HOST} --port ${SERVER_PORT} --device_url ${DEVICE_URL} --device_root ${DEVICE_ROOT} --data_cache_path /data/mmmu_data
"
echo '>>>>>>>>>>>> Test mmmu over. <<<<<<<<<<<<';

set +x
