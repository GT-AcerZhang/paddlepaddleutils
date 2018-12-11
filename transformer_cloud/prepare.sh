#!/bin/bash
set -e
rm -f brpc/*.whl grpc/*.whl brpcrdma/*.whl
mkdir -p brpc grpc brpcrdma

#grpc
#cp ~/go/src/github.com/PaddlePaddle/Paddle/build/build_1.0.0_centos6u3_release_gpu/python/dist/paddlepaddle_gpu-0.9.0-cp27-cp27mu-linux_x86_64.whl ./grpc 
#cp ~/go/src/github.com/PaddlePaddle/Paddle/build/build_0b3d8fc_centos6u3_release_gpu/python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl ./grpc 
#cp ~/go/src/github.com/PaddlePaddle/Paddle/build/build_1.1.0_centos6u3_release_gpu/python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl ./grpc 

#brpc
cp /ssd1/gongwb/go/src/github.com/PaddlePaddle/Paddle/build/build_brpcserial_centos6u3_release_gpu/python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl ./brpc


#rdma
#cp ~/go/src/github.com/PaddlePaddle/Paddle/build/build_brpcserial_rdma_centos6u3_release_gpu/python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl ./brpcrdma

#models
#cp ~/go/src/github.com/PaddlePaddle/models/fluid/PaddleNLP/neural_machine_translation/transformer/*.py transformer/

