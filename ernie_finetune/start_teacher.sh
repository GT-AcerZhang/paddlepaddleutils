unset https_proxy http_proxy
export CUDA_VISIBLE_DEVICES=4
export LD_LIBRARY_PATH=/root/go/soft/env/cuda-9.0/lib64:/root/go/soft/cuda10-cudnn7.6.5.32/lib64:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
python3.6 -m paddle_serving_server_gpu.serve \
  --model ./ernie_senti_server/ \
  --port 19293 \
  --thread 8 \
  --mem_optim \
  --gpu_ids 0


