#!/bin/bash
set -e

rm -f whl/*.whl

#whl
#cp ~/go/src/github.com/PaddlePaddle/Paddle/build/build_1.1.0_Release_gpu/python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl whl/
#cp ~/go/src/github.com/PaddlePaddle/Paddle/build/build_nccl2test_RelWithDebInfo_gpu/python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl whl/
cp ~/go/src/github.com/PaddlePaddle/Paddle/build/build_exectest_RelWithDebInfo_gpu/python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl whl/
#cp ~/go/src/github.com/PaddlePaddle/Paddle/build/build_fcf370e_RelWithDebInfo_gpu/python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl whl/
#cp ~/go/src/github.com/PaddlePaddle/Paddle/build/build_develop_RelWithDebInfo_gpu/python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl whl/
cp ~/go/src/github.com/PaddlePaddle/Paddle/build/third_party/RelWithDebInfo_gpu/install/mkldnn/lib/libmkldnn.so.0 .

#models
cp ~/go/src/github.com/PaddlePaddle/models/fluid/PaddleNLP/neural_machine_translation/transformer/*.py models/
