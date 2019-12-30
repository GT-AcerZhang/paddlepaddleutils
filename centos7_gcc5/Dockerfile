FROM paddlepaddle/paddle_manylinux_devel:cuda10.0_cudnn7

RUN yum install -y yum-utils
RUN yum -y install centos-release-scl
RUN yum groups install -y "Development Tools"

RUN yum install -y zlib-devel bzip2-devel ncurses-devel sqlite-devel readline-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel libffi-devel libtool which
RUN yum install -y glibc-devel libstdc++-devel glib2-devel libX11-devel libXext-devel libXrender-devel  mesa-libGL-devel libICE-devel libSM-devel ncurses-devel freetype-devel libpng-devel wget

#ssl
RUN cd /usr/src && rm -f openssl-1.1.1.tar.gz  && \
   wget https://www.openssl.org/source/old/1.1.1/openssl-1.1.1.tar.gz && \
   tar -xzvf openssl-1.1.1.tar.gz && cd openssl-1.1.1 && \
   ./config no-ssl2 no-shared -fPIC --prefix=/usr/local/ssl && \
    make -j &&  make install

#curl
RUN cd /usr/src/ && rm -f curl-7.49.1.tar.gz && \
	wget https://curl.askapache.com/curl-7.49.1.tar.gz && \
	tar -xzvf curl-7.49.1.tar.gz && cd curl-7.49.1 && \
	LIBS="-ldl -pthread" ./configure --with-ssl=/usr/local/ssl --disable-shared && \
	make -j && make install

#bin utils
RUN cd /usr/src && rm -f binutils-2.27.tar.gz && \
	wget https://ftp.gnu.org/gnu/binutils/binutils-2.27.tar.gz && \
        tar xzf binutils-2.27.tar.gz && cd binutils-2.27 && \
        ./configure --prefix=/opt/rh/devtoolset-2/root/usr/ --enable-64-bit-archive && make -j `nproc` && make install

#protobuf
RUN cd /usr/src &&  rm -f protobuf-cpp-3.1.0.tar.gz && \
    wget -q --no-check-certificate https://github.com/protocolbuffers/protobuf/releases/download/v3.1.0/protobuf-cpp-3.1.0.tar.gz && \
    tar xzf protobuf-cpp-3.1.0.tar.gz && \
    cd protobuf-3.1.0 && ./configure && make -j && make install && cd .. && rm -f protobuf-cpp-3.1.0.tar.gz

#patchelf
RUN unset https_proxy http_proxy
RUN cd /usr/src/ && wget http://download-ib01.fedoraproject.org/pub/epel/7/x86_64/Packages/p/patchelf-0.9-10.el7.x86_64.rpm && \
       rpm -Uvh patchelf-0.9-10.el7.x86_64.rpm

RUN export PATH=/usr/local/bin:$PATH
RUN export LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib:$LD_LIBRARY_PATH

RUN cd /usr/src

# Download and extract source code
EVN gcc_version="5.4.0"
RUN rm -rf gcc-${gcc_version}_build gcc-${gcc_version}
RUN rm -f gcc-${gcc_version}.tar.bz2 gmp-6.1.0.tar.xz mpc-1.0.3.tar.gz mpfr-3.1.5.tar.xz

RUN wget --no-verbose --no-check-certificate \
        http://ftpmirror.gnu.org/gcc/gcc-${gcc_version}/gcc-${gcc_version}.tar.bz2 && \
    tar xf gcc-${gcc_version}.tar.bz2

RUN wget --no-verbose --no-check-certificate \
        https://gmplib.org/download/gmp/gmp-6.1.0.tar.xz && \
    tar xf gmp-6.1.0.tar.xz && \
    mv gmp-6.1.0 gcc-${gcc_version}/gmp

RUN wget --no-verbose --no-check-certificate \
        ftp://ftp.gnu.org/gnu/mpc/mpc-1.0.3.tar.gz && \
    tar xf mpc-1.0.3.tar.gz && \
    mv mpc-1.0.3 gcc-${gcc_version}/mpc

RUN wget --no-verbose --no-check-certificate \
        https://www.mpfr.org/mpfr-3.1.5/mpfr-3.1.5.tar.xz && \
    tar xf mpfr-3.1.5.tar.xz && \
    mv mpfr-3.1.5 gcc-${gcc_version}/mpfr

# Compile and install GCC
# "we highly recommend that GCC be built into a separate directory from the sources which does not reside within the source tree"
RUN mkdir gcc-${gcc_version}_build && \
    cd gcc-${gcc_version}_build && \
    ../gcc-${gcc_version}/configure \
        --prefix=/opt/compiler/gcc540 \
        --disable-multilib \
        --enable-languages=c,c++ \
        --enable-libstdcxx-threads \
        --enable-libstdcxx-time \
        --enable-shared \
        --enable-__cxa_atexit \
        --disable-libunwind-exceptions \
        --disable-libada \
        --host x86_64-redhat-linux-gnu \
        --build x86_64-redhat-linux-gnu \
        --with-default-libstdcxx-abi=gcc4-compatible && \
	make clean && make -j  && make install

RUn ln -s /opt/compiler/gcc540/bin/gcc /opt/compiler/gcc540/bin/cc
ENV PATH /opt/compiler/gcc540/bin/:$PATH
ENV LD_LIBRARY_PATH /opt/compiler/gcc540/lib:/opt/compiler/gcc540/lib64:${LD_LIBRARY_PATH}

#cmake
RUN cd /usr/src && rm -f cmake-3.5.2.tar.gz && \
	wget -q https://cmake.org/files/v3.5/cmake-3.5.2.tar.gz && tar xzf cmake-3.5.2.tar.gz && \
	cd cmake-3.5.2 && ./bootstrap && \
	make -j && make install && cd .. && rm  -f cmake-3.5.2.tar.gz

#compile python37
RUN cd /usr/src && wget https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tgz && \
	tar xzf Python-3.7.0.tgz && cd Python-3.7.0 && \
        CFLAGS="-Wformat" ./configure --prefix=/opt/python/python3.7.0 --with-openssl=/usr/local/ssl --enable-shared && \
	make -j && make altinstall

RUN rm -f /usr/bin/python3 /usr/bin/pip3 && ln -s /opt/python/python3.7.0/bin/python3.7 /usr/bin/python3 && \
	 ln -s /opt/python/python3.7.0/bin/pip3.7 /usr/bin/pip3 && \
	 echo "/opt/python/python3.7.0/lib" > /etc/ld.so.conf.d/python3.7.0.conf && \
	 ldconfig

RUN pip3 install  opencv-python wheel

RUN cp ./lib/cuda10-cudnn7.6.5.32/include/cudnn.h  /usr/local/cuda/include/
RUN cp ./lib/cuda10-cudnn7.6.5.32/lib64/libcudnn.so  /lib64/
RUN cp ./lib/nccl_2.3.5/include/nccl.h /usr/local/cuda/include/
RUN cp ./lib/soft/nccl_2.3.5/lib/libnccl.so /lib64/





