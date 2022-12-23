#!/bin/bash

##proxy is not a valid URL
#./bazel clean --expunge

if [ $# -eq 0 ]; then
    echo "build.sh [vqindex_api | vqindex_py]"
    exit
fi

bazel_name="bazel_linux"
os_name=`uname -a` 
if [[ $os_name =~ "Darwin" ]];then
    bazel_name="bazel_mac"
elif [[ $os_name =~ "Linux" ]];then
    bazel_name="bazel_linux"
else
    echo "Only support Linux and MacOS"
    exit
fi

command -v clang >/dev/null 2>&1 || { echo >&2 "I require clang but it's not installed.  Aborting."; exit 1; }

python3 configure.py

if [ "$1" = "vqindex_py" ]
then
    python_bin_path=`which python3`
    if [ -z "$python_bin_path" ]
    then
        echo "python3 is not exist."
        exit
    fi

    CC=clang ./${bazel_name} build -c opt --features=thin_lto --copt=-mavx2 --copt=-mfma --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w //scann/scann_ops/cc:vqindex_py --action_env PYTHON_BIN_PATH=${python_bin_path}
    cp -rf bazel-bin/scann/scann_ops/cc/libvqindex_py.so test/vqindex_py.so
    cp -rf bazel-bin/external/local_config_tf/libtensorflow_framework.* test/

    echo "Build suss, you can use test/test.py to test."
    exit
fi

if [ "$1" = "vqindex_api" ]
then
    if [ ! -d "libs" ];then
       mkdir libs
    fi

    CC=clang ./${bazel_name} build -c opt --features=thin_lto --copt=-mavx2 --copt=-mfma --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w //scann/scann_ops/cc:vqindex_api
    cp -rf scann/scann_ops/cc/vqindex_api.h libs/vqindex_api.h
    cp -rf bazel-bin/scann/scann_ops/cc/libvqindex_api.so libs/libvqindex_api.so
    cp -rf bazel-bin/external/local_config_tf/libtensorflow_framework.* libs/

    echo "Build libs/libvqindex_api.so suss."
    exit
fi
