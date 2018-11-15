#!/bin/bash

if [[ $# != 1 ]]; then
    echo "Please input jobname"
    exit 1
fi

HGCP_CLIENR_BIN=~/.hgcp/software-install/HGCP_client/bin

#export ${AFSSERVER}... here.
source ~/.hgcp/user.sh

${HGCP_CLIENR_BIN}/submit \
        --hdfs ${AFSSERVER} \
        --hdfs-user ${AFSUSER} \
        --hdfs-passwd ${AFSPWD} \
        --hdfs-path ${AFSPATH} \
        --file-dir ./ \
        --job-name $1 \
        --num-nodes 2 \
        --queue-name yq01-p40-4-8 \
        --num-task-pernode 1 \
        --gpu-pnode 8 \
        --job-script job.sh
