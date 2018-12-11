#!/bin/bash
set -x

source env.sh

output_path=/app/inf/mpi/bml-guest/paddlepaddle/gongweibao/transformer/en-fr/output 
output_path2=/app/inf/mpi/bml-guest/paddlepaddle/gongweibao/transformer/en-de/output

${HADOOP_BIN} fs -Dhadoop.job.ugi=${HADOOP_UGI} -fs ${HADOOP_FS} -rmr ${output_path} ${output_path2}
