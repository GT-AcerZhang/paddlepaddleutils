FROM hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda10.0-cudnn7-dev
RUN rm /etc/apt/sources.list.d/cuda.list

ARG UBUNTU_MIRROR
RUN /bin/bash -c 'if [[ -n ${UBUNTU_MIRROR} ]]; then sed -i 's#http://archive.ubuntu.com/ubuntu#${UBUNTU_MIRROR}#g' /etc/apt/sources.list; fi'

RUN unset https_proxy http_proxy
ENV http_proxy=http://172.19.57.45:3128
ENV https_proxy=http://172.19.57.45:3128
RUN rm -rf /var/lib/apt/lists/*
#RUN cat /etc/apt/apt.conf

RUN apt-get update && apt-get -y install ctags build-essential cmake python-dev llvm \
    man manpages  manpages-de manpages-de-dev manpages-dev lrzsz cscope gdb clang-format \
    iputils-ping python2.7-dbg libgoogle-perftools-dev inetutils-traceroute itop sysstat


RUN cd /root && \
    git clone https://github.com/vim/vim.git && \
    cd vim && \
    ./configure --with-features=huge \
            --enable-multibyte \
            --enable-rubyinterp=yes \
            --enable-pythoninterp=yes \
            --enable-perlinterp=yes \
            --enable-luainterp=yes \
	    --enable-python3interp=yes  \
            --enable-gui=gtk2 --enable-cscope --prefix=/usr && \
    make -j && make install
            
RUN cd /root && \
    wget --quiet https://raw.githubusercontent.com/gongweibao/paddlepaddleutils/develop/build/.vimrc && \
    git clone https://github.com/VundleVim/Vundle.vim.git .vim/bundle/Vundle.vim

ENV GOROOT=/usr/local/go GOPATH=/root/go
ENV PATH=$PATH:${GOROOT}/bin
RUN cd /root && \
    git clone https://github.com/Valloric/YouCompleteMe.git .vim/bundle/YouCompleteMe && \
    cd .vim/bundle/YouCompleteMe && \
    git submodule update --init --recursive && \
    ./install.py --gocode-completer --clang-completer
    
#RUN vim +PluginInstall +qall
#RUN vim +GoInstallBinaries +qall

RUN python -m pip install paddlepaddle-gpu
RUN python3.6 -m pip install paddlepaddle-gpu sklearn paddlehub

RUN echo "alias vi=vim" >> /root/.bashrc
ENV LD_LIBRARY_PATH=/root/go/soft/cuda10-cudnn7.6.5.32/lib64:${LD_LIBRARY_PATH}:/usr/lib64/:/usr/local/lib/

RUN git config --global user.email "weibao.gong@gmail.com" && \
    git config --global user.name "gongweibao"

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
