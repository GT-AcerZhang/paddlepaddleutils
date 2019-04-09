#!/bin/bash
set -e
mkdir -p whl
rm -f whl/*.whl

cp \
/ssd2/gongwb/go/src/github.com/PaddlePaddle/Paddle/build/build_cent_chengduo16393_release_gpu_y_grpc/python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl \
./whl/


