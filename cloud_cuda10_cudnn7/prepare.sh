#!/bin/bash
set -e
mkdir -p whl
rm -f whl/*.whl

whl_path=/ssd2/gongwb/go/src/github.com/PaddlePaddle/Paddle/build/build_cent_develop_release_gpu_y_grpc/

cp  ${whl_path}/python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl ./whl/


