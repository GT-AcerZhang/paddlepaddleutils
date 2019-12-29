#!/bin/bash
set -e
rpc_image=registry.baidu.com/paddlepaddle-public/paddle_manylinux_devel_dali:cud10_cudnn7

docker build --network host . -t ${rpc_image}

docker push ${rpc_image}

