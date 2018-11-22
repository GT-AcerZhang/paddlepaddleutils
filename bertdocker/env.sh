#!/bin/bash 
set -xe

export iplist=192.168.16.29,192.168.16.30

#http_proxy
unset http_proxy
unset https_proxy

#paddle envs
export PADDLE_PSERVER_PORT=9184
export PADDLE_TRAINER_IPS=${iplist} 
export PADDLE_CURRENT_IP=`hostname -i`

iparray=(${iplist//,/ })
for i in "${!iparray[@]}"; do
if [ ${iparray[$i]} == ${PADDLE_CURRENT_IP} ]; then
    export PADDLE_TRAINER_ID=$i
fi
done

export TRAINING_ROLE=TRAINER
export PADDLE_INIT_TRAINER_COUNT=${#iparray[@]}
export PADDLE_PORT=${PADDLE_PSERVER_PORT}
export PADDLE_TRAINERS=${PADDLE_TRAINER_IPS}
export POD_IP=${PADDLE_CURRENT_IP}
export PADDLE_TRAINERS_NUM=${PADDLE_INIT_TRAINER_COUNT}
export PADDLE_IS_LOCAL=0
echo "****************************************************"

#paddle debug envs
export GLOG_v=100
export GLOG_logtostderr=1

#nccl debug envs
export NCCL_DEBUG=VERSION
#export NCCL_IB_DISABLE=1
#export NCCL_IB_GDR_LEVEL=4
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth2

