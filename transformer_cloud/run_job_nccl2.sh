#!/bin/bash

set -x

#source /home/gongwb/application/bin/paddlecloud-cli/demo/command-test/config.sh 
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
    --k8s-memory 100Gi \
    --distribute-job-type NCCL2 \
    --job-name gongweibao-transformer-nccl2-en-fr \
    --start-cmd "python -u run.py --update_method nccl2" \
    --job-conf transformer/conf.py \
    --files transformer/config.py transformer/infer.py transformer/model.py transformer/optim.py transformer/profile.py transformer/reader.py transformer/train.py transformer/util.py transformer/run.py \
    --k8s-not-local  \
    --k8s-trainers 2 \
    --k8s-cpu-cores 7 \
    --image-addr "registry.baidu.com/gongweibao/distributed_paddle:nccl2"
