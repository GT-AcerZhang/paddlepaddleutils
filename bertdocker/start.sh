#!/bin/bash
set -e
image=bootstrapper:5000/gongweibao/distributed_paddle_ubuntu_grpc:1.1.0

docker pull ${image}

export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')

docker run  ${CUDA_SO} ${DEVICES} -it --rm --network=host \
    -v /etc/libibverbs.d:/etc/libibverbs.d \
    -v /usr/local/nvidia/lib64:/usr/local/nvidia/lib64 \
    -v /usr/local/rdma/lib64:/usr/local/rdma/lib64 \
    -v /usr/lib64/mlnx_ofed/valgrind:/usr/lib64/mlnx_ofed/valgrind \
    -v /root/gongwb/bert:/root/gongwb/bert ${image}
