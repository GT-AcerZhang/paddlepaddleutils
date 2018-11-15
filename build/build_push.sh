#!/bin/bash

image=registry.baidu.com/gongweibao/paddle-dev:cuda8-cudnn7

docker build --build-arg UBUNTU_MIRROR=http://ftp.sjtu.edu.cn/ubuntu/ . -t ${image}
docker push ${image}
