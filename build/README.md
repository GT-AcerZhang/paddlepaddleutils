## 构建镜像
```
docker built . -t gongweibao/paddledev
docker push gongweibao/paddledev
```

## Host机器上的准备
- 创建自己的用户

```
useradd -d /home/gongwb -m gongwb  
chown gongwb -R /home/gongwb
usermod -a -G root gongwb
passwd gongwb

su gongwb
mkdir -p ~/go/src/github.com/PaddlePaddle
cd ~/go/src/github.com/PaddlePaddle
git clone https://github.com/gongweibao/Paddle.git
```

## 启动镜像
`--security-opt seccomp=unconfined`是为了能够使用gdb调试，看这里：http://blog.csdn.net/so_dota_so/article/details/77509530

```
nvidia-docker run -d -p 2202:22 --name gwbpaddle  --security-opt seccomp=unconfined -it -v ~/go/src/github.com/PaddlePaddle/Paddle:/paddle -v ~/go:/root/go  gongweibao/paddledev
```

如果nvidia-docker有问题，那么采用不推荐的更古老的办法：
```
export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
docker run ${CUDA_SO} ${DEVICES} -d -p 2202:22 --name gwbpaddle --network=host  --security-opt seccomp=unconfined -it -v ~/go/src/github.com/PaddlePaddle/Paddle:/paddle -v ~/go:/root/go  gongweibao/paddledev /bin/bash 
```

## 登陆镜像
```
ssh -p 2202 root@host
```

第一次需要:

```
cd ~/go/src/github.com/PaddlePaddle/Paddle
pre-commit install
vim +GoInstallBinaries +qall
```

## 常用的paddle命令

- 编译安装所有

```
export WITH_GPU=OFF
export WITH_AVX=ON
export WITH_TEST=ON
export WITH_GOLANG=ON
export WITH_TESTING=ON
export RUN_TEST=ON
cd /paddle
bash paddle/scripts/docker/build.sh
```

- 定制cmake

```
cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_DOC=OFF \
      -DWITH_GPU=${WITH_GPU:-OFF} \
      -DWITH_AVX=${WITH_AVX:-ON} \
      -DWITH_GOLANG=${WITH_GOLANG:-ON} \
      -DWITH_SWIG_PY=ON \
      -DCUDNN_ROOT=/usr/ \
      -DWITH_STYLE_CHECK=${WITH_STYLE_CHECK:-OFF} \
      -DWITH_TESTING=${WITH_TESTING:-ON} \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make xxxx
make install
```

- [修改python路径](https://www.douban.com/note/334738164/)

可以通过这种方式而不用安装所有的东西达到进行测试部分的目的
```
echo "/root/go/src/github.com/PaddlePaddle/Paddle/python" > /usr/local/lib/python2.7/dist-packages/paddle.pth
```

## pre-commit  
```
pip install -U pre-commit
cd paddle
pre-commit install
pre-commit run -a
git commti -a -m 'Update pre-commit changes'
git push
```

## install glide
```
apt-get install software-properties-common
add-apt-repository ppa:masterminds/glide 
apt-get update
apt-get install glide
```


## paddlepaddle 国内镜像
```
hub.baidubce.com/paddlepaddle/paddle
```
