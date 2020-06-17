export CUDA_VISIBLE_DEVICES=4
python3.6 -m paddle_serving_server_gpu.serve \
  --model ./ernie_senti_server/ \
  --port 19293 \
  --thread 8 \
  --mem_optim true \
  --gpu_ids 0


