#!/bin/bash

function clean() {
    set +e
    rm -rf /qnn
}

set -e
trap clean EXIT

bash /lib/qnn/unzip_qnn.sh
source /qnn/bin/envsetup.sh
ANDROID_NDK=/ndk
echo -e "\033[32mSetting up NDK environment variable\033[0m"
if [ -z "$ANDROID_NDK" ]; then
    echo -e "\033[31mNDK not found\033[0m"
    exit 1
else
    echo -e "\033[32mNDK found at $ANDROID_NDK\033[0m"
fi

if [ -z "$QNN_SDK_ROOT" ]; then
    echo -e "\033[31mQNN_SDK_ROOT not found\033[0m"
    exit 1
else
    echo -e "\033[32mQNN_SDK_ROOT found at $QNN_SDK_ROOT\033[0m"
fi

cd /code

echo -e "\033[32mCreating build directory for Android\033[0m"
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DPOWERSERVE_WITH_QNN=ON -DPOWERSERVE_ENABLE_HTPRPCPOLL=ON -DPOWERSERVE_ENABLE_HMXPWRCFG=ON -DPOWERSERVE_USE_DUMMY=ON -S . -B build_android

echo -e "\033[32mBuilding project for Android\033[0m"
cmake --build build_android --config RelWithDebInfo --parallel 12 --target all

tempmodel1="/tempA"
tempmodel2="/tempB"
mkdir -p "${tempmodel1}"
mkdir -p "${tempmodel2}"
touch "${tempmodel1}/model.json"
touch "${tempmodel2}/model.json"

./powerserve create --only-extract-qnn -m "${tempmodel1}" -d "${tempmodel2}" --exe-path /code/build_android/out

rm -rf "${tempmodel1}" || true
rm -rf "${tempmodel2}" || true

chmod -R 777 /code/proj
chmod -R 777 /code/build_android
chmod 777 /models
chmod 777 *.log
