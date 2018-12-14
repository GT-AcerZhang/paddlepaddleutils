#!/bin/bash
set -xe

rm -f whl/*.whl

#whl
cp /ssd1/gongwb/go/src/github.com/PaddlePaddle/Paddle/build/build_bertslow_RelWithDebInfo_gpu/python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl whl/
cp /ssd1/gongwb/go/src/github.com/PaddlePaddle/Paddle/build/third_party/RelWithDebInfo_gpu/install/mkldnn/lib/libmkldnn.so.0 .

#models
#cp /ssd1/gongwb/go/src/gitlab.com/bert/*.py ./models/
#cp /ssd1/gongwb/go/src/gitlab.com/bert/*.conf ./models/
#cp /ssd1/gongwb/go/src/gitlab.com/bert/shell/* ./shell/
