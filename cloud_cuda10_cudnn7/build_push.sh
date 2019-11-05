#!/bin/bash
set -e
rpc_image=registry.baidu.com/gongweibao/distributed_paddle:cud10_cudnn7

docker build --network host . -t ${rpc_image}

docker push ${rpc_image}

