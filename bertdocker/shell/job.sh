#!/bin/bash
set -x

mpirun hostname

source ./model_conf
mpirun sh ./shell/setup.sh

iplist=`cat nodelist-${SLURM_JOB_ID} | xargs  | sed 's/ /,/g'`
mpirun --bind-to none -x iplist=${iplist} sh shell/train.sh
