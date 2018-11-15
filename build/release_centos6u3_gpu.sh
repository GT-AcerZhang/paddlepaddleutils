#!/bin/bash
set -x 
if [[ $# != 1 ]]; then
    echo "build branch"
    exit 1
fi

export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.11-ucs4/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.11-ucs2/lib:}
export PATH=/opt/python/cp27-cp27mu/bin/:${PATH}
PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/python/cp27-cp27mu/bin/python"

INFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR:-/root/.cache/inference_demo}

WITH_ANAKIN=OFF
WITH_DOC=OFF
WITH_GPU=ON
WITH_DISTRIBUTE=ON
WITH_SWIG_PY=ON
WITH_PYTHON=ON
WITH_FLUID_ONLY=ON

build_dir=build_$1_centos6u3_release_gpu
mkdir -p  ${build_dir}
cd ${build_dir}

cmake ../../ -DTHIRD_PARTY_PATH=/paddle/build/third_party/centos6u3_release_gpu/ \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release} \
        ${PYTHON_FLAGS} \
        -DWITH_DSO=ON \
        -DWITH_DOC=${WITH_DOC:-OFF} \
        -DWITH_GPU=${WITH_GPU:-OFF} \
        -DWITH_AMD_GPU=${WITH_AMD_GPU:-OFF} \
        -DWITH_DISTRIBUTE=${WITH_DISTRIBUTE:-OFF} \
        -DWITH_MKL=${WITH_MKL:-ON} \
        -DWITH_AVX=${WITH_AVX:-OFF} \
        -DWITH_GOLANG=${WITH_GOLANG:-OFF} \
        -DCUDA_ARCH_NAME=${CUDA_ARCH_NAME:-All} \
        -DWITH_SWIG_PY=${WITH_SWIG_PY:-ON} \
        -DWITH_C_API=${WITH_C_API:-OFF} \
        -DWITH_PYTHON=${WITH_PYTHON:-ON} \
        -DCUDNN_ROOT=/usr/ \
        -DWITH_TESTING=${WITH_TESTING:-ON} \
        -DWITH_FAST_BUNDLE_TEST=ON \
        -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake \
        -DWITH_FLUID_ONLY=${WITH_FLUID_ONLY:-OFF} \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DWITH_CONTRIB=${WITH_CONTRIB:-ON} \
        -DWITH_INFERENCE=${WITH_INFERENCE:-ON} \
        -DWITH_INFERENCE_API_TEST=${WITH_INFERENCE_API_TEST:-ON} \
        -DINFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR} \
        -DWITH_ANAKIN=${WITH_ANAKIN:-OFF} \
        -DPY_VERSION=${PY_VERSION:-2.7}

export http_proxy=http://172.19.56.198:8123
export https_proxy=http://172.19.56.198:8123
export no_proxy="localhost,127.0.0.1,localaddress,.localdomain.com,paddle-inference-dist.cdn.bcebos.com"

make -j 25
