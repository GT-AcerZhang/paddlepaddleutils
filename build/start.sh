#!/bin/bash
set -xe

export https_proxy=http://172.19.57.45:3128/
export http_proxy=http://172.19.57.45:3128/
export no_proxy="localhost,127.0.0.1,localaddress,.localdomain.com,.cdn.bcebos.com,.baidu.com"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=/root/go/soft/cuda10-cudnn7.6.5.32/lib64:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/:/usr/lib/x86_64-linux-gnu/

os=ubuntu
if grep --quiet "Ubuntu" /etc/issue; then
  os=ubuntu
  export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
elif grep --quiet "CentOS" /etc/issue; then
  os=cent

  #export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.11-ucs4/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.11-ucs2/lib:}
  #export PATH=/opt/python/cp27-cp27mu/bin/:${PATH}
  #PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/python/cp27-cp27mu/bin/python" 
  export JAVA_HOME=/usr/lib/jvm/java-1.7.0-openjdk.x86_64
else
  echo "can't find os, exit"
  exit 1
fi

#arr=("v1.8.2.1" "v1.7.1.1" "v1.6.3.1" "v1.6.1.1")
arr=("v1.7.1.1" "v1.6.3.1" "v1.6.1.1")
#arr=("v1.6.3.1" "v1.6.1.1")
#arr=("v1.6.1.1")
for v in ${arr[@]} ; do
    git checkout $v
    git pull origin $v

    echo "processing:"$v 
    git rev-parse HEAD

    for py_v in "2.7" "3.5.1" "3.6.0" "3.7.0" ; do
        #rm -rf third_party/${os}_${v}_release_gpu_${py_v}/inference_demo/

        slim_path=build_${os}_${v}_release_gpu_y_grpc_${py_v}/python/paddle/fluid/contrib/slim/tests
        rm -rf ${slim_path}/checkpoints_uniform_restore/ ${slim_path}/checkpoints_uniform_restore_tmp/

        #rm -rf build_${os}_${v}_release_gpu_y_grpc_${py_v}

        paddle_version=`echo ${v} |  cut -c 2-6`
        export PADDLE_VERSION=${paddle_version}

        ./build.sh -py_v $py_v

        #pushd build_${os}_${v}_release_gpu_y_grpc_2.7
        #set +e
        #../paddle_build.sh test
        #set -e
        #popd

    done

done
