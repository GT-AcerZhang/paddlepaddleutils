FROM registry.baidu.com/paddlecloud/paddlecloud-runenv-centos6u3-online:paddlecloud-gcc482-cuda8.0_cudnn7
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>
ARG rpc_name
USER root
ENV HOME /root
RUN yum install -y openssl-devel
ADD ./${rpc_name}/*.whl /tmp/
ADD pip.conf /etc/pip.conf
ADD pip.conf ~/.pip/pip.conf
RUN pip uninstall -y requests; exit 0
RUN pip uninstall -y matplotlib; exit 0
RUN pip uninstall -y numpy; exit 0
#RUN rm -f /etc/pip.conf
#RUN mkdir -p /home/paddlepaddle/lib && cp /home/opt/gcc-4.8.2.bpkg-r2/gcc-4.8.2.bpkg-r2/lib64/libssl.so.1.0.0 /home/opt/gcc-4.8.2.bpkg-r2/gcc-4.8.2.bpkg-r2/lib64/libcrypto.so.1.0.0 /home/paddlepaddle/lib
#ENV LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v5/cuda/lib64/:/home/work/cuda-8.0/lib64/:/usr/local/lib/:/nccl/lib:$LD_LIBRARY_PATH:/home/paddlepaddle/lib
ENV LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v5/cuda/lib64/:/home/work/cuda-8.0/lib64/:/usr/local/lib/:/nccl/lib:$LD_LIBRARY_PATH
#/home/opt/gcc-4.8.2.bpkg-r2/gcc-4.8.2.bpkg-r2/lib64/
# run paddle version to install python packages first
RUN pip uninstall -y paddlepaddle
RUN pip --default-timeout=100 install /tmp/*.whl --index-url=http://pip.baidu.com/root/baidu/ --trusted-host pip.baidu.com 
#RUN pip install /tmp/*.whl
RUN rm -f /tmp/*.whl && python -c "import paddle" && ldconfig
