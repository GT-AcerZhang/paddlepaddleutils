#!/bin/bash

unset GREP_OPTIONS
branch=`git branch | grep \* | cut -d ' ' -f2`
place=gpu
build_type=debug
rpc=grpc
testing=y

while true ; do
  case "$1" in
    -b) branch="$2" ; shift 2 ;;
    -p) place="$2" ; shift 2 ;;
    -t) testing="$2" ; shift 2 ;;
    -c) build_type="$2" ; shift 2 ;;
    -r) rpc="$2" ; shift 2 ;;
    *) 
       if [[ ${#1} > 0 ]]; then
          echo "not supported arugments ${1}" ; exit 1 ;
       else
           break
       fi 
       ;;
  esac
done


echo "branch=${branch}"
echo "place=${place}"
echo "build_type=${build_type}"
echo "rpc=${rpc}"
echo "testing=${testing}"

#[[ $place == "gpu" || $place == "cpu" ]] || echo "not support ${place}" ; exit 1 ;
#[[ $testing == "yes" || $testing == "no" ]] || echo "not support ${testing}" ; exit 1 ;
#[[ $build_type == "release" || $build_type == "debug" ]] || echo "not support ${build_type}" ; exit 1 ;
#[[ $rpc == "brpc" || $rpc == "grpc" || $rpc == "brpcrdma" ]] || echo "not support ${rpc}" ; exit 1 ;

echo "last"

case "$place" in
    gpu) WITH_GPU=ON ;;
    cpu) WITH_GPU=OFF ;;
    *) echo "not support ${place}" ; exit 1 ;;
esac

case "$testing" in
    y) WITH_TESTING=ON ;;
    n) WITH_TESTING=OFF ;;
    *) echo "not support ${testing}" ; exit 1 ;;
esac

case "$build_type" in
    release) CMAKE_BUILD_TYPE=Release ;;
    debug) CMAKE_BUILD_TYPE=RelWithDebInfo ;;
    *) echo "not support ${build_type}" ; exit 1 ;;
esac

case "$rpc" in
    brpc) WITH_GRPC=OFF ;;
    grpc) WITH_GRPC=ON ;;
    rdma) WITH_GRPC=OFF ; WITH_BRPC_RDMA=ON ;;
    *) echo "not support ${rpc}" ; exit 1 ;;
esac

build_dir=build_${branch}_${build_type}_${place}_${testing}_${rpc}
mkdir -p  ${build_dir}
cd ${build_dir}

set -x
cmake ../../  -DTHIRD_PARTY_PATH=/paddle/build/third_party/${build_type}_${place}/ \
         -DWITH_MKLML=ON \
         -DWITH_MKLDNN=ON \
         -DWITH_GPU=${WITH_GPU:-ON} \
         -DWITH_C_API=OFF \
         -DWITH_DISTRIBUTE=ON \
         -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-RelWithDebInfo} \
         -DWITH_TESTING=${WITH_TESTING:-ON} \
         -DWITH_STYLE_CHECK=OFF \
         -DWITH_CONTRIB=OFF \
         -DWITH_GRPC=${WITH_GRPC:-ON} \
         -DWITH_BRPC_RDMA=${WITH_BRPC_RDMA:-OFF} \
         -DWITH_FLUID_ONLY=ON \
         -DCMAKE_INSTALL_PREFIX=/root/paddlebuild/${build_type}_${place}/install \
         -DCMAKE_EXPORT_COMPILE_COMMANDS=ON


