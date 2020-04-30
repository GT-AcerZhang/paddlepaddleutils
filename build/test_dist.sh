#!/bin/bash
set -xe

#export http_proxy=http://172.19.56.198:8123
#export https_proxy=http://172.19.56.198:8123
#export no_proxy="localhost,127.0.0.1,localaddress,.localdomain.com"
#unset http_proxy https_proxy

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/:/workspace/brpc
export FLAGS_fraction_of_gpu_memory_to_use=0.15
export PYTHONPATH=$PYTHONPATH:$(pwd)/python

#python ./python/paddle/fluid/tests/unittests/test_desc_clone.py
#python ./python/paddle/fluid/tests/unittests/test_dist_transpiler.py

#export DIST_CHECK_ERROR_LOG=ON
unset http_proxy https_proxy

for i in {1..2}; do 
    echo $i
    pushd ./python/paddle/fluid/tests/unittests/
    python test_dist_se_resnext.py
    python test_dist_word2vec.py
    python test_listen_and_serv_op.py
    python test_dist_mnist.py
    python test_dist_train.py
    python test_dist_simnet_bow.py
    #python test_dist_transformer.py
    popd
done

