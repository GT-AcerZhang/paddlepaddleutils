#!/bin/bash
if [[ $# != 3 ]]; then
    echo "build branch [release|debug] [gpu|cpu]"
    exit 1
fi

build_type=Release
if [[ $2 == "debug" ]];  then
    build_type=RelWithDebInfo
fi

place=gpu
WITH_GPU=ON
if [[ $3 == "cpu" ]]; then
    place=cpu
    WITH_GPU=OFF
fi

build_dir=build_$1_${build_type}_${place}

mkdir -p  ${build_dir}
cd ${build_dir}

WITH_GRPC=ON
WITH_CONTRIB=OFF
WITH_BRPC_RDMA=OFF
if [[ $1 == *"brpc"* ]]; then
    WITH_GRPC=OFF
    WITH_CONTRIB=OFF
fi

if [[ $1 == *"rdma"* ]]; then
    WITH_GRPC=OFF
    WITH_CONTRIB=OFF
    WITH_BRPC_RDMA=ON
fi

set -x
cmake ../../  -DTHIRD_PARTY_PATH=/paddle/build/third_party/${build_type}_${place}/ \
         -DWITH_MKLML=ON \
         -DWITH_MKLDNN=ON \
         -DWITH_GPU=${WITH_GPU} \
         -DWITH_C_API=OFF \
         -DWITH_DISTRIBUTE=ON \
         -DCMAKE_BUILD_TYPE=${build_type} \
         -DWITH_TESTING=ON \
         -DWITH_STYLE_CHECK=OFF \
         -DWITH_CONTRIB=${WITH_CONTRIB} \
         -DWITH_GRPC=${WITH_GRPC} \
         -DWITH_BRPC_RDMA=${WITH_BRPC_RDMA} \
         -DWITH_FLUID_ONLY=OFF \
         -DWITH_INFERENCE=ON \
         -DCMAKE_INSTALL_PREFIX=/root/paddlebuild/${build_type}_${place}/install
