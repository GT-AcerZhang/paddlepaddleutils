export CUDA_VISIBLE_DEVICES=0,1,2,3 #,4,5,6,7 #,6,7 #0,1,2,3 #6,7
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/nvidia/lib64:/usr/local/rdma/lib64:/usr/lib64/mlnx_ofed/valgrind:$LD_LIBRARY_PATH
export FLAGS_fraction_of_gpu_memory_to_use=1.0
export TRAINING_ROLE=TRAINER
export PADDLE_PSERVERS=127.0.0.1
export PADDLE_IS_LOCAL=0
export PADDLE_INIT_TRAINER_COUNT=1
export PADDLE_TRAINER_ID=1
export PADDLE_PORT=30314
export PADDLE_TRAINERS=192.168.16.29,192.168.16.30
export POD_IP=192.168.16.29
export PADDLE_TRAINERS_NUM=2
export GLOG_logtostderr=1
export GLOG_v=3
#export NCCL_SOCKET_IFNAME=eth0
#export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1

cd /workspace/

python -u train.py \
    --src_vocab_fpath 'cluster_test_data_en_fr/thirdparty/vocab.wordpiece.en-fr' \
    --trg_vocab_fpath 'cluster_test_data_en_fr/thirdparty/vocab.wordpiece.en-fr' \
    --special_token '<s>' '<e>' '<unk>'  \
    --token_delimiter '\x01' \
    --train_file_pattern 'cluster_test_data_en_fr/train/train.wordpiece.en-fr.0' \
    --val_file_pattern 'cluster_test_data_en_fr/thirdparty/newstest2014.wordpiece.en-fr' \
    --use_token_batch True \
    --batch_size  3200 \
    --sort_type pool \
    --pool_size 200000 \
    --local False \
  --shuffle True \
  --shuffle_batch True \
  --use_py_reader True \
  --use_mem_opt True \
  --enable_ce False \
  --update_method nccl2 \
  d_model 1024 \
  d_inner_hid 4096 \
  n_head 16 \
  learning_rate 2.0 \
  warmup_steps 8000 \
  beta2 0.997 \
  pass_num 100 \
  prepostprocess_dropout 0.3 \
  attention_dropout 0.1 \
  relu_dropout 0.1 \
  weight_sharing True
