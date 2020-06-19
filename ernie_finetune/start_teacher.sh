set -ex
unset https_proxy http_proxy
export LD_LIBRARY_PATH=/root/go/soft/env/cuda-9.0/lib64:/root/go/soft/cuda10-cudnn7.6.5.32/lib64:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
for i in {0..7}
do
    #mkdir -p teacher/dir_$i
    #export CUDA_VISIBLE_DEVICES=$i
    nohup python3.6 -m paddle_serving_server_gpu.serve \
      --model ./ernie_senti_server/ \
      --port 1929$i \
      --thread 8 \
      --mem_optim \
      --gpu_ids $i > teacher/$i.log 2>&1 &
done
