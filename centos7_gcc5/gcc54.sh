#!/bin/bash
set -xe

cd /usr/src

# Download and extract source code
export gcc_version="5.4.0"
rm -rf gcc-${gcc_version}_build gcc-${gcc_version}
rm -f gcc-${gcc_version}.tar.bz2 gmp-6.1.0.tar.xz mpc-1.0.3.tar.gz mpfr-3.1.5.tar.xz

wget --no-verbose --no-check-certificate \
        http://ftpmirror.gnu.org/gcc/gcc-${gcc_version}/gcc-${gcc_version}.tar.bz2 && \
    tar xf gcc-${gcc_version}.tar.bz2

wget --no-verbose --no-check-certificate \
        https://gmplib.org/download/gmp/gmp-6.1.0.tar.xz && \
    tar xf gmp-6.1.0.tar.xz && \
    mv gmp-6.1.0 gcc-${gcc_version}/gmp

wget --no-verbose --no-check-certificate \
        ftp://ftp.gnu.org/gnu/mpc/mpc-1.0.3.tar.gz && \
    tar xf mpc-1.0.3.tar.gz && \
    mv mpc-1.0.3 gcc-${gcc_version}/mpc

wget --no-verbose --no-check-certificate \
        https://www.mpfr.org/mpfr-3.1.5/mpfr-3.1.5.tar.xz && \
    tar xf mpfr-3.1.5.tar.xz && \
    mv mpfr-3.1.5 gcc-${gcc_version}/mpfr

# Compile and install GCC
# "we highly recommend that GCC be built into a separate directory from the sources which does not reside within the source tree"
mkdir gcc-${gcc_version}_build && \
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

ln -s /opt/compiler/gcc540/bin/gcc /opt/compiler/gcc540/bin/cc
export PATH=/opt/compiler/gcc540/bin/:$PATH
export LD_LIBRARY_PATH=/opt/compiler/gcc540/lib:${LD_LIBRARY_PATH}

#compile python37
#CFLAGS="-Wformat" ./configure --enable-optimizations --enable-shared && \ 
cd /usr/src && wget https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tgz \
	tar xzf Python-3.7.0.tgz && cd Python-3.7.0 && \
        CFLAGS="-Wformat" ./configure --prefix=/opt/python/python3.7.0 --with-openssl=/usr/local/ssl --enable-shared && \
	make -j && make altinstall

ln -s /opt/python/python3.5/bin/python3 /usr/bin/python3
echo "/opt/python/python3.7.0" > /etc/ld.so.conf.d/python3.7.conf
ldconfig

pip3 install -y  python-opencv

#install pip
#cd /usr/src && curl -O https://bootstrap.pypa.io/get-pip.py && \
#	python3 get-pip.py && \
#	rm -rf get-pip.py && \
#	ln -s /usr/local/bin/python3.7 /usr/local/bin/python3 && \
#	ln -s /usr/local/bin/python3.7 /usr/local/bin/python
