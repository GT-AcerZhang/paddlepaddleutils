FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# Use UBUNTU_MIRROR can speed up apt-get speed.
ARG UBUNTU_MIRROR
RUN /bin/bash -c 'if [[ -n ${UBUNTU_MIRROR} ]]; then sed -i 's#http://archive.ubuntu.com/ubuntu#${UBUNTU_MIRROR}#g' /etc/apt/sources.list; fi'

#RUN cat /etc/resolv.conf

RUN apt-get update && apt-get install -y python python-pip iputils-ping libgtk2.0-dev wget vim net-tools iftop python-opencv gdb python2.7-dbg
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/lib/libcudnn.so && ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/libnccl.so

# IMPORTANT:
# Add "ENV http_proxy=http://ip:port" if your download is slow, and don't forget to unset it at runtime.
# exmaple: unset http_proxy && unset https_proxy && python fluid_benchmark.py ...

ENV http_proxy=http://172.19.57.45:3128
ENV https_proxy=http://172.19.57.45:3128

RUN pip install -U pip

ADD requirements.txt /paddle/
RUN pip install  -r /paddle/requirements.txt

RUN mkdir /workspace
ADD ./data/ /workspace/data

ADD whl/*.whl /
RUN pip install /*.whl && rm -f /*.whl 
ADD libmkldnn.so.0 /usr/local/lib

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/:/usr/local/nvidia/lib64:/usr/local/rdma/lib64:/root/gongwb/transformer/lib:/usr/local/lib/python2.7/dist-packages/paddle/libs:$LD_LIBRARY_PATH
ADD models/ /workspace/
ADD train.sh env.sh /workspace/

