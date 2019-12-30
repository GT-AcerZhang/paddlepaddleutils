#!/bin/bash
set -xe

yum install -y yum-utils
yum -y install centos-release-scl
yum groups install -y "Development Tools"

yum install -y zlib-devel bzip2-devel ncurses-devel sqlite-devel readline-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel libffi-devel libtool
yum install -y glibc-devel libstdc++-devel glib2-devel libX11-devel libXext-devel libXrender-devel  mesa-libGL-devel libICE-devel libSM-devel ncurses-devel freetype-devel libpng-devel wget

#ssl
cd /usr/src && rm -f openssl-1.1.1.tar.gz  && \
   wget https://www.openssl.org/source/old/1.1.1/openssl-1.1.1.tar.gz && \
   tar -xzvf openssl-1.1.1.tar.gz && cd openssl-1.1.1 && \
   ./config no-ssl2 no-shared -fPIC --prefix=/usr/local/ssl && \
    make -j &&  make install

#curl
cd /usr/src/ && rm -f curl-7.49.1.tar.gz && \
	wget https://curl.askapache.com/curl-7.49.1.tar.gz && \
	tar -xzvf curl-7.49.1.tar.gz && cd curl-7.49.1 && \
	LIBS="-ldl -pthread" ./configure --with-ssl=/usr/local/ssl --disable-shared && \
	make -j && make install

#bin utils
cd /usr/src && rm -f binutils-2.27.tar.gz && \
	wget https://ftp.gnu.org/gnu/binutils/binutils-2.27.tar.gz && \
        tar xzf binutils-2.27.tar.gz && cd binutils-2.27 && \
        ./configure --prefix=/opt/rh/devtoolset-2/root/usr/ --enable-64-bit-archive && make -j `nproc` && make install

#cmake
cd /usr/src && rm -f cmake-3.5.2.tar.gz && \
	wget -q https://cmake.org/files/v3.5/cmake-3.5.2.tar.gz && tar xzf cmake-3.5.2.tar.gz && \
	cd cmake-3.5.2 && ./bootstrap && \
	make -j && make install && cd .. && rm  -f cmake-3.5.2.tar.gz

#protobuf
cd /usr/src &&  rm -f protobuf-cpp-3.1.0.tar.gz && \
    wget -q --no-check-certificate https://github.com/protocolbuffers/protobuf/releases/download/v3.1.0/protobuf-cpp-3.1.0.tar.gz && \
    tar xzf protobuf-cpp-3.1.0.tar.gz && \
    cd protobuf-3.1.0 && ./configure && make -j && make install && cd .. && rm -f protobuf-cpp-3.1.0.tar.gz

#patchelf
#cd /usr/src && wget -q  --no-check-certificate http://nipy.bic.berkeley.edu/manylinux/patchelf-0.9njs2.tar.gz &&  \
#	tar -xzf patchelf-0.9njs2.tar.gz &&  \
#	 cd patchelf-0.9njs2 && ./configure && make -j && make install
unset https_proxy http_proxy
cd /usr/src/ && wget http://download-ib01.fedoraproject.org/pub/epel/7/x86_64/Packages/p/patchelf-0.9-10.el7.x86_64.rpm \
       rpm -Uvh patchelf-0.9-10.el7.x86_64.rpm

export PATH=/usr/local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib:$LD_LIBRARY_PATH
	
#cd /usr/src && wget -q --no-check-certificate ftp://ftp.gnu.org/gnu/make/make-4.1.tar.gz && tar make-4.1.tar.gz \
#	cd make-4.1 && ./configure && make -j && make install
