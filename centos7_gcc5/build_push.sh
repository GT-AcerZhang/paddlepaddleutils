#!/bin/bash
set -e
rpc_image=registry.baidu.com/paddlepaddle-public/paddle_manylinux_cent7_gcc540_devel:cud10_cudnn7

docker build --network host . -t ${rpc_image}

docker push ${rpc_image}

