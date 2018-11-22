#!/bin/bash

set -e

mkdir -p ../whl
rm -f ../whl/*.whl

#cp ~/go/src/github.com/PaddlePaddle/Paddle/build/build_exectest_centos6u3_release_gpu/python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl ../whl/
cp ~/go/src/github.com/PaddlePaddle/Paddle/build/build_bertgather_centos6u3_release_gpu/python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl ../whl/
cp ~/go/src/github.com/PaddlePaddle/models/fluid/PaddleNLP/neural_machine_translation/transformer/*.py ./models
