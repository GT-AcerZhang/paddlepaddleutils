FROM paddlepaddle/paddle:latest-dev-cuda10.2-cudnn7-gcc82
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

RUN rm -f /usr/bin/python /usr/local/bin/python && ln -s /usr/local/bin/python3.7 /usr/bin/python
RUN rm -f /usr/bin/pip /usr/local/bin/pip && ln -s /usr/local/bin/pip3.7 /usr/bin/pip
RUN pip install --upgrade pip

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

RUN wget -qO- https://golang.org/dl/go1.13.15.linux-amd64.tar.gz | \
	tar -xz -C /usr/local && rm -rf /root/gopath && \
	mkdir /root/gopath && \
    	mkdir /root/gopath/bin && \
	mkdir /root/gopath/src

ENV GOROOT=/usr/local/go GOPATH=/root/gopath
# should not be in the same line with GOROOT definition, otherwise docker build could not find GOROOT.	
ENV PATH=${PATH}:${GOROOT}/bin:${GOPATH}/bin
    
RUN cd /root && \
    wget --quiet https://raw.githubusercontent.com/gongweibao/paddlepaddleutils/develop/build/.vimrc && \
    git clone https://github.com/VundleVim/Vundle.vim.git .vim/bundle/Vundle.vim

RUN rm -f /usr/bin/python3 /usr/local/bin/python3 && ln -s /usr/local/bin/python3.7 /usr/bin/python3
RUN rm -f /usr/bin/pip3 /usr/local/bin/pip3 && ln -s /usr/local/bin/pip3.7 /usr/bin/pip3

ENV GOROOT=/usr/local/go GOPATH=/root/go
ENV PATH=/usr/bin/:/usr/local/bin:$PATH:${GOROOT}/bin
RUN cd /root && \
    git clone https://github.com/Valloric/YouCompleteMe.git .vim/bundle/YouCompleteMe && \
    cd .vim/bundle/YouCompleteMe && \
    git submodule update --init --recursive && \
    ./install.py --gocode-completer --clang-completer
    
#RUN vim +PluginInstall +qall
#RUN vim +GoInstallBinaries +qall

RUN cd /root && wget --quiet https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-installer-linux-x86_64.sh && \
    chmod +x bazel-3.1.0-installer-linux-x86_64.sh && \
    ./bazel-3.1.0-installer-linux-x86_64.sh --user 

RUN echo "alias vi=vim" >> /root/.bashrc
ENV PATH=$PATH:$HOME/bin
RUN echo "export PATH=$PATH" >> /root/.bashrc

ENV LD_LIBRARY_PATH=/root/go/soft/cuda10-cudnn7.6.5.32/lib64:${LD_LIBRARY_PATH}:/usr/lib64/:/usr/local/lib/

RUN git config --global user.email "weibao.gong@gmail.com" && \
    git config --global user.name "gongweibao"

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
