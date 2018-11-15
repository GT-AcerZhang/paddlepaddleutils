#!/bin/bash
set -xe

python ./python/paddle/fluid/tests/unittests/test_desc_clone.py

python ./python/paddle/fluid/tests/unittests/test_dist_transpiler.py

pushd ./python/paddle/fluid/tests/unittests/
python test_dist_se_resnext.py
popd

python ./python/paddle/fluid/tests/unittests/test_dist_word2vec.py
python ./python/paddle/fluid/tests/unittests/test_listen_and_serv_op.py
python ./python/paddle/fluid/tests/unittests/test_dist_mnist.py
python ./python/paddle/fluid/tests/unittests/test_dist_train.py

pushd ./python/paddle/fluid/tests/unittests/
python test_dist_transformer.py
popd

