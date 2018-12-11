#!/bin/bash

set -x

if [[ -z $1 ]]; then
    echo "input rpc name"
    exit 1
fi

rpc_name=$1

update_method="nccl2"
if [[ $rpc_name == "grpc" || $rpc_name == "brpc" ]]; then
    update_method="pserver"
fi

source /home/gongwb/local/etc/paddlecloud-cli/config.sh

k8s_wall_time="96:00:00"

paddlecloud job \
    --server ${server} \
    --port ${port} \
    --user-ak ${user_ak} \
    --user-sk ${user_sk} \
    train --job-version paddle-fluid-custom \
    --k8s-gpu-type baidu/gpu_p40 \
    --cluster-name paddle-jpaas-ai00-gpu \
    --k8s-gpu-cards 4 \
    --k8s-priority high \
    --k8s-wall-time ${k8s_wall_time} \
    --k8s-memory 120Gi \
    --k8s-ps-memory 40Gi \
    --job-name gongweibao-${rpc_name}-transformer \
    --start-cmd "python -u run.py --update_method ${update_method}" \
    --job-conf transformer/conf.py \
    --files transformer/config.py transformer/infer.py transformer/model.py transformer/optim.py transformer/profile.py transformer/reader.py transformer/train.py transformer/util.py transformer/run.py \
    --k8s-not-local  \
    --k8s-trainers 2 \
    --k8s-cpu-cores 7 \
    --k8s-ps-num 2 \
    --k8s-ps-cores 7 \
    --image-addr "registry.baidu.com/gongweibao/distributed_paddle:${rpc_name}"
