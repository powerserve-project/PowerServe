#!/bin/bash

source ./.gitlab/common.sh

temp_disable_errexit try_twice 20 "$@"
