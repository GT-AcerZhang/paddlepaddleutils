FROM paddlepaddle/paddle_manylinux_devel:cuda10.0_cudnn7

ADD ./install_base.sh ./gcc54.sh /root
RUN cd root && bash install_base.sh && rm -f install_base.sh
RUN cd root && bash gcc54.sh && rm -f gcc54.sh
RUN bash ./cp_cudnn.sh




