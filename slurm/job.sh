#!/bin/bash
set -x

mpirun hostname

iplist=`cat nodelist-${SLURM_JOB_ID} | xargs  | sed 's/ /,/g'`
mpirun --bind-to none -x iplist=${iplist} sh test_transformer_nccl.sh
