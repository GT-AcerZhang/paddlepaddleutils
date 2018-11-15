FROM paddlepaddle/paddle:latest-dev

ARG UBUNTU_MIRROR
RUN /bin/bash -c 'if [[ -n ${UBUNTU_MIRROR} ]]; then sed -i 's#http://archive.ubuntu.com/ubuntu#${UBUNTU_MIRROR}#g' /etc/apt/sources.list; fi'

RUN apt-get update && apt-get -y install ctags build-essential cmake python-dev llvm   python3-dev \
    man manpages  manpages-de manpages-de-dev manpages-dev lrzsz cscope gdb clang-format \
    iputils-ping python2.7-dbg libgoogle-perftools-dev inetutils-traceroute itop sysstat

RUN cd /root && \
    git clone https://github.com/vim/vim.git && \
    cd vim && \
    ./configure --with-features=huge \
            --enable-multibyte \
            --enable-rubyinterp=yes \
            --enable-pythoninterp=yes \
            --with-python-config-dir=/usr/lib/python2.7/config \
            --enable-perlinterp=yes \
            --enable-luainterp=yes \
            --enable-gui=gtk2 --enable-cscope --prefix=/usr && \
    make -j 8 && make install
            
RUN cd /root && \
    wget --quiet https://raw.githubusercontent.com/gongweibao/tests/develop/paddle/.vimrc && \
    git clone https://github.com/VundleVim/Vundle.vim.git .vim/bundle/Vundle.vim

ENV http_proxy=http://172.19.56.198:8123
ENV https_proxy=http://172.19.56.198:8123

RUN cd /root && \
    git clone https://github.com/Valloric/YouCompleteMe.git .vim/bundle/YouCompleteMe && \
    cd .vim/bundle/YouCompleteMe && \
    git checkout 7ba7eef && \
    git submodule update --init --recursive && \
    ./install.py --gocode-completer --clang-completer
    
ENV GOROOT=/usr/local/go GOPATH=/root/go
RUN mkdir -p /root/go/bin
ENV PATH=${PATH}:${GOROOT}/bin:${GOPATH}/bin
RUN curl -s -q https://glide.sh/get | sh
RUN go get -u github.com/alecthomas/gometalinter 
RUN gometalinter --install
RUN go get github.com/google/pprof

RUN pip  install nltk scipy cpplint yep

#RUN mkdir -p /root/go/src/github.com/PaddlePaddle 
#
#RUN vim +PluginInstall +qall
#RUN vim +GoInstallBinaries +qall

RUN echo "alias vi=vim" >> /root/.bashrc
RUN echo "export GOROOT=/usr/local/go" >> /root/.bashrc
RUN echo "export GOPATH=/root/go" >> /root/.bashrc
RUN echo "export PATH=${PATH}:${GOPATH}/bin" >> /root/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs/:${LD_LIBRARY_PATH} >> /root/.bashrc"

RUN git config --global user.email "weibao.gong@gmail.com" && \
    git config --global user.name "gongweibao"

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
