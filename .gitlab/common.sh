#!/bin/bash

function print_info() {
    msg=("$@")

    if [[ "$TERM" == *"color"* ]]; then
        echo -e "\033[32m${msg[@]}\033[0m"
    else
        echo "${msg[@]}"
    fi
}

function print_error() {
    msg=("$@")

    if [[ "$TERM" == *"color"* ]]; then
        echo -e "\033[31m${msg[@]}\033[0m"
    else
        echo "${msg[@]}"
    fi
}

function print_warning() {
    msg=("$@")

    if [[ "$TERM" == *"color"* ]]; then
        echo -e "\033[33m${msg[@]}\033[0m"
    else
        echo "${msg[@]}"
    fi
}

function run_to() {
    # set timeout for cmd
    time_limit=$1
    shift
    cmd=("$@")

    print_info "[timeout=${time_limit}]> ${cmd[@]}"

    timeout ${time_limit} "${cmd[@]}"
    ret=$?
    if [ "${ret}" -ne "0" ]; then
        print_error "Timeout Error (ret=${ret}): ${cmd[@]}"
    fi
    return ${ret}
}

function try_twice() {
    time_limit=$1
    shift
    cmd=("$@")

    run_to ${time_limit} "${cmd[@]}"
    ret=$?

    if [ "${ret}" -ne "0" ]; then
        print_warning "Try Again ..."
        sleep 2
        run_to $((time_limit * 2)) "${cmd[@]}"
    fi
    ret=$?

    return ${ret}
}

function temp_disable_errexit() {
    cmd=("$@")

    if set -o | grep -q "errexit.*on"; then
        errexit_was_on=true
    else
        errexit_was_on=false
    fi

    set +e
    "${cmd[@]}"
    cmd_ret=$?

    if [ "$errexit_was_on" = true ]; then
        set -e
    fi
    return ${cmd_ret}
}

function clean_workspace() {
    DEVICE_ROOT=$1
    DEVICE_URL=$2
    DEVICE_PORT=$3
    WORKSPACE_PATH=$4
    BIN_PATH=$5
    QNN_PATH=$6
    SDK_PATH=$7

    temp_disable_errexit try_twice 10 ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
        cd ${WORKSPACE_PATH};
        echo '>>>>>>>>>>>> Cleaning files. <<<<<<<<<<<<';
        ls -Alh;
        if [ \$(ls | wc -l) -gt 0 ]; then
            rm -r *;
        else
            echo 'No files to clean.';
        fi;
        echo '>>>>>>>>>>>> Cleaning over. <<<<<<<<<<<<';
        ls -Alh;
    "

    ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
        cd ${WORKSPACE_PATH};
        echo '>>>>>>>>>>>> Copying files. <<<<<<<<<<<<';
        cp -r ${QNN_PATH}/* .;
        cp -r ${SDK_PATH}/* .;
        echo '>>>>>>>>>>>> Copying over. <<<<<<<<<<<<';
        ls -Alh;
        if [ \$(ls | grep mixed_layers.json | wc -l) -lt 1 ]; then
            echo 'No Mixed Layers Info.';
        else
            cat mixed_layers.json;
        fi
    "
}
