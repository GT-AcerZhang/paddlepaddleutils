#!/bin/bash
set -xe

export https_proxy=http://172.19.57.45:3128/
export http_proxy=http://172.19.57.45:3128/
export no_proxy="localhost,127.0.0.1,localaddress,.localdomain.com,.cdn.bcebos.com,.baidu.com"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=/root/go/soft/cuda10-cudnn7.6.5.32/lib64:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/:/usr/lib/x86_64-linux-gnu/
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

os=ubuntu
if grep --quiet "Ubuntu" /etc/issue; then
  os=ubuntu
elif grep --quiet "CentOS" /etc/issue; then
  os=cent
else
  echo "can't find os, exit"
  exit 1
fi

arr=("v1.8.2.1" "v1.7.1.1" "v1.6.3.1" "v1.6.1.1")
#arr=("v1.7.1.1" "v1.6.3.1" "v1.6.1.1")
#arr=("v1.6.3.1" "v1.6.1.1")
#arr=("v1.6.1.1")
for v in ${arr[@]} ; do
    git checkout $v
    git pull origin $v

    echo "processing:"$v 
    git rev-parse HEAD

    rm -rf third_party/${os}_${v}_release_gpu_2.7/inference_demo/

    slim_path=build_${os}_${v}_release_gpu_y_grpc_2.7/python/paddle/fluid/contrib/slim/tests
    rm -rf ${slim_path}/checkpoints_uniform_restore/ ${slim_path}/checkpoints_uniform_restore_tmp/

    rm -rf build_${os}_${v}_release_gpu_y_grpc_2.7

    paddle_version=`echo ${v} |  cut -c 2-6`
    export PADDLE_VERSION=${paddle_version}

    ./build.sh

    pushd build_${os}_${v}_release_gpu_y_grpc_2.7
    #ctest -V
    set +e
    ../paddle_build.sh test
    set -e
    popd
done
