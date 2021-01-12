#!/bin/bash
set -x
LOCAL_ASCEND=/usr/local/Ascend

export PATH=$PATH:${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/ 

export LD_LIBRARY_PATH=${LOCAL_ASCEND}/add-ons/:${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe/op_tiling:/usr/local/Ascend/driver/lib64/common/:${LD_LIBRARY_PATH}:${LOCAL_ASCEND}/fwkacllib/lib64/

g++ --version

g++ -I/usr/local/Ascend/fwkacllib/include -L/usr/local/Ascend/fwkacllib/lib64 -lgraph -lge_runner -o test_init test_init.cc
