#!/bin/bash
rpc_image=registry.baidu.com/paddlepaddle-public/distributed_paddle_dgc:centos6u3-cuda8-cudnn7
docker build --network host . -t ${rpc_image}
docker push ${rpc_image}
