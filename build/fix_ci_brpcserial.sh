#!/bin/bash 
set -e 

mkdir -p fix_brpcserial_ci
pushd fix_brpcserial_ci

cmake ../../ -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DTHIRD_PARTY_PATH=/paddle/build/third_party/RelWithDebInfo_gpu/ \
	-DWITH_DSO=ON \
	-DWITH_DOC=OFF \
	-DWITH_GPU=ON \
	-DWITH_AMD_GPU=OFF \
	-DWITH_DISTRIBUTE=ON \
	-DWITH_MKL=ON \
	-DWITH_AVX=ON \
	-DWITH_GOLANG=OFF \
	-DCUDA_ARCH_NAME=Auto \
	-DWITH_SWIG_PY=ON \
	-DWITH_C_API=OFF \
	-DWITH_PYTHON=ON \
	-DCUDNN_ROOT=/usr/ \
	-DWITH_TESTING=ON \
	-DWITH_FAST_BUNDLE_TEST=ON \
	-DWITH_FLUID_ONLY=ON \
	-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DWITH_CONTRIB=ON \
	-DWITH_INFERENCE=ON \
	-DWITH_INFERENCE_API_TEST=ON \
	-DINFERENCE_DEMO_INSTALL_DIR=/root/.cache/inference_demo \
	-DWITH_ANAKIN=OFF \
	-DPY_VERSION=2.7 \
	-DCMAKE_INSTALL_PREFIX=/paddle/build

make test_trt_models -j 25

popd
