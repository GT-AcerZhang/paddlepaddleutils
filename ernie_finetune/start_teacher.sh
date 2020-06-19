unset https_proxy http_proxy
export CUDA_VISIBLE_DEVICES=4
export LD_LIBRARY_PATH=/root/go/soft/env/cuda-9.0/lib64:/root/go/soft/cuda10-cudnn7.6.5.32/lib64:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
nohup python3.6 -m paddle_serving_server_gpu.serve \
  --model ./ernie_senti_server/ \
  --port 19294 \
  --thread 8 \
  --mem_optim \
  --gpu_ids 0 > 4.log  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python3.6 -m paddle_serving_server_gpu.serve \
  --model ./ernie_senti_server/ \
  --port 19295 \
  --thread 8 \
  --mem_optim \
  --gpu_ids 0 > 5.log 2>&1 &


export CUDA_VISIBLE_DEVICES=6
nohup python3.6 -m paddle_serving_server_gpu.serve \
  --model ./ernie_senti_server/ \
  --port 19296 \
  --thread 8 \
  --mem_optim \
  --gpu_ids 0 > 6.log 2>&1 &

