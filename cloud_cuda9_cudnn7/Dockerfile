FROM registry.baidu.com/paddlecloud/base-images:paddlecloud-gcc482-cuda9.0_cudnn7
USER root
ENV HOME /root
RUN yum install -y openssl-devel wondershaper
ADD ./whl/*.whl /tmp/
ADD pip.conf /etc/pip.conf
ADD pip.conf ~/.pip/pip.conf
RUN pip uninstall -y requests; exit 0
RUN pip uninstall -y matplotlib; exit 0
RUN pip uninstall -y numpy; exit 0
ENV LD_LIBRARY_PATH=/home/work/cuda-9.0/lib64/:/home/work/cudnn/cudnn_v7/cuda/lib64/:/usr/local/lib/:/nccl/lib:$LD_LIBRARY_PATH

# run paddle version to install python packages first
RUN pip uninstall -y paddlepaddle paddlepaddle-gpu
RUN pip --default-timeout=100 install /tmp/*.whl --index-url=http://pip.baidu.com/root/baidu/ --trusted-host pip.baidu.com 
RUN pip install /tmp/*.whl
RUN rm -f /tmp/*.whl && python -c "import paddle"
