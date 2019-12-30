FROM paddlepaddle/paddle_manylinux_devel:cuda10.0_cudnn7

RUN yum install -y yum-utils
RUN yum install -y centos-release-scl https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm

RUN yum install -y zlib-devel bzip2-devel ncurses-devel sqlite-devel readline-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel libffi-devel

RUN yum install -y gcc gcc-c++ wget xz libgcc glibc-devel glibc-headers \
   zlib-devel bzip2-devel ncurses-devel sqlite-devel readline-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel gpg libffi-devel \
   glibc-devel libstdc++-devel glib2-devel libX11-devel libXext-devel libXrender-devel  mesa-libGL-devel libICE-devel libSM-devel ncurses-devel

RUN yum -y install \
    automake \
    autoconf \
    libtool \
    autotools-latest \
    bison \
    bzip2 \
    patchelf \
    cmake28 \
    curl \
    diffutils \
    expat-devel \
    gettext \
    git \
    kernel-devel-`uname -r` \
    file \
    make \
    nasm \
    openssl \
    openssl-devel \
    patch \
    unzip \
    which \
    yasm

# Run some tests
RUN gcc --version && \
    g++ --version && \
    which gcc && \
    which g++
RUN mkdir ~/tests && \
    cd ~/tests && \
    echo '#include <iostream>' > main.cpp && \
    echo 'using namespace std;' >> main.cpp && \
    echo 'int main() {' >> main.cpp && \
    echo '    cout << "Hello world!" << endl;' >> main.cpp && \
    echo '    return 0;' >> main.cpp && \
    echo '}' >> main.cpp && \
    g++ main.cpp -o main && \
    ./main

# Download and compile GCC-5
# http://en.librehat.com/blog/build-gcc-5-dot-2-on-rhel-6/

# Download and extract source code
ENV gcc_version "5.4.0"
RUN wget --no-verbose \
        http://ftpmirror.gnu.org/gcc/gcc-${gcc_version}/gcc-${gcc_version}.tar.bz2 && \
    tar xf gcc-${gcc_version}.tar.bz2
RUN wget --no-verbose \
        https://gmplib.org/download/gmp/gmp-6.1.0.tar.xz && \
    tar xf gmp-6.1.0.tar.xz && \
    mv gmp-6.1.0 gcc-${gcc_version}/gmp
RUN wget --no-verbose \
        ftp://ftp.gnu.org/gnu/mpc/mpc-1.0.3.tar.gz && \
    tar xf mpc-1.0.3.tar.gz && \
    mv mpc-1.0.3 gcc-${gcc_version}/mpc
RUN wget --no-verbose \
        https://www.mpfr.org/mpfr-3.1.5/mpfr-3.1.5.tar.xz && \
    tar xf mpfr-3.1.5.tar.xz && \
    mv mpfr-3.1.5 gcc-${gcc_version}/mpfr

#ENV PATH /usr/local/go/bin:/root/gopath/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Compile and install GCC
# "we highly recommend that GCC be built into a separate directory from the sources which does not reside within the source tree"
RUN mkdir gcc-${gcc_version}_build && \
    cd gcc-${gcc_version}_build && \
    ../gcc-${gcc_version}/configure \
        --prefix=/usr \
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
        --with-default-libstdcxx-abi=gcc4-compatible
RUN cd gcc-${gcc_version}_build && make -j 20
RUN cd gcc-${gcc_version}_build && make install

# Validate the installed compiler
RUN hash -r && \
    gcc --version && \
    g++ --version && \
    which gcc && \
    which g++

# Register new libraries with `ldconfig`
RUN echo "/usr/local/lib64" > usrLocalLib64.conf && \
    mv usrLocalLib64.conf /etc/ld.so.conf.d/ && \
    ldconfig


# Clean out all the garbage
RUN rm -rf ~/${gcc_release} ~/{gcc_release}_build ~/tests

#compile python37
RUN cd /usr/src && \
	wget https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tgz && \
	tar xzf Python-3.7.0.tgz && \
	cd Python-3.7.0 && \
	CFLAGS="-Wformat" ./configure --enable-optimizations --enable-shared && \
	make altinstall

#install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python3.7 get-pip.py && \
	rm -rf get-pip.py && \
	ln -s /usr/local/bin/python3.7 /usr/local/bin/python3 && \
	ln -s /usr/local/bin/python3.7 /usr/local/bin/python

RUN wget https://ftp.gnu.org/gnu/binutils/binutils-2.27.tar.gz && \
	tar xzf binutils-2.27.tar.gz && cd binutils-2.27 && \
	./configure --prefix=/opt/rh/devtoolset-2/root/usr/ --enable-64-bit-archive && make -j `nproc` && make install
