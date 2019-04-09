#!/bin/bash
rpc_image=registry.baidu.com/gongweibao/distributed_paddle_dgc:cud9_cudnn7
docker build --network host . -t ${rpc_image}
docker push ${rpc_image}

