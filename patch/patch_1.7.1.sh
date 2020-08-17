set -e
#path="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop"
path="https://raw.githubusercontent.com/gongweibao/Paddle/v1.8.2.1"
#shell.cc
wget ${path}/paddle/fluid/framework/io/shell.cc -O paddle/fluid/framework/io/shell.cc
wget ${path}/paddle/fluid/framework/io/shell.h -O paddle/fluid/framework/io/shell.h
wget ${path}/paddle/fluid/platform/port.h -O paddle/fluid/platform/port.h
wget ${path}/paddle/fluid/platform/timer.h -O paddle/fluid/platform/timer.h
wget ${path}/paddle/fluid/platform/timer.cc -O paddle/fluid/platform/timer.cc


# fswrapper
if [[ -f python/paddle/distributed/fs_wrapper.py ]]; then
    git rm python/paddle/distributed/fs_wrapper.py
fi

# fswrapper
if [[ -f python/paddle/fluid/incubate/fleet/utils/hdfs.py ]]; then
    git rm python/paddle/fluid/incubate/fleet/utils/hdfs.py
fi

# checkpoint and unittest
mkdir -p python/paddle/fluid/incubate/checkpoint
touch python/paddle/fluid/incubate/checkpoint/__init__.py 
wget ${path}/python/paddle/fluid/incubate/checkpoint/auto_checkpoint.py -O python/paddle/fluid/incubate/checkpoint/auto_checkpoint.py
wget ${path}/python/paddle/fluid/incubate/checkpoint/checkpoint_saver.py -O python/paddle/fluid/incubate/checkpoint/checkpoint_saver.py
wget ${path}/python/paddle/fluid/tests/unittests/test_fleet_checkpoint.py -O python/paddle/fluid/tests/unittests/test_fleet_checkpoint.py
wget ${path}/python/paddle/fluid/tests/unittests/test_auto_checkpoint.py -O python/paddle/fluid/tests/unittests/test_auto_checkpoint.py
wget ${path}/python/paddle/fluid/tests/unittests/test_auto_checkpoint2.py -O python/paddle/fluid/tests/unittests/test_auto_checkpoint2.py

#gwb_path="https://raw.githubusercontent.com/gongweibao/Paddle/1483b31a9a200cd819c73cacf65399cac8b93307"
gwb_path="https://raw.githubusercontent.com/gongweibao/Paddle/v1.8.2.1"
wget ${gwb_path}/python/paddle/fluid/tests/unittests/test_dataloader_auto_checkpoint.py -O python/paddle/fluid/tests/unittests/test_dataloader_auto_checkpoint.py
wget ${gwb_path}/python/paddle/fluid/tests/unittests/test_dataloader_auto_checkpoint2.py -O python/paddle/fluid/tests/unittests/test_dataloader_auto_checkpoint2.py
wget ${gwb_path}/python/paddle/fluid/incubate/checkpoint/dataloader_auto_checkpoint.py -O python/paddle/fluid/incubate/checkpoint/dataloader_auto_checkpoint.py
wget ${gwb_path}/python/paddle/fluid/incubate/checkpoint/test_checkpoint_saver.py.py -O python/paddle/fluid/incubate/checkpoint/test_checkpoint_saver.py
wget ${gwb_path}/python/paddle/fluid/tests/unittests/auto_checkpoint_utils.py -O python/paddle/fluid/tests/unittests/auto_checkpoint_utils.py

# hdfs
wget ${path}/python/paddle/fluid/tests/unittests/test_fs_interface.py -O python/paddle/fluid/tests/unittests/test_fs_interface.py
wget ${path}/python/paddle/fluid/tests/unittests/test_hdfs.py -O python/paddle/fluid/tests/unittests/test_hdfs.py
mkdir -p python/paddle/fluid/incubate/fleet/utils
touch python/paddle/fluid/incubate/fleet/utils/__init_.py
wget ${path}/python/paddle/fluid/incubate/fleet/utils/fs.py -O ./python/paddle/fluid/incubate/fleet/utils/fs.py
echo "complete"
