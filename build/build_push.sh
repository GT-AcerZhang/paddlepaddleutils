#!/bin/bash

image=registry.baidu.com/gongweibao/paddle-dev:cuda10.2-cudnn7-gcc82

docker build --pull --network host --build-arg UBUNTU_MIRROR=http://ftp.sjtu.edu.cn/ubuntu/  . -t ${image}
#docker build  --network host . -t ${image}
docker push ${image}
