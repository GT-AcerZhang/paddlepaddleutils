#!/bin/bash
set -xe

source ./user.sh

if [[ ! -d "cluster_test_data_en_fr" ]]; then
    hadoop fs -Dhadoop.job.ugi=${mulan_hdfs_ugi} \
        -fs ${mulan_hdfs_server} \
        -get /app/inf/mpi/bml-guest/paddlepaddle/public/transformer/cluster_test_data_en_fr .
fi

#source ./env.sh

export PADDLE_IS_LOCAL=0
export FLAGS_fraction_of_gpu_memory_to_use=1.0

echo $LD_LIBRARY_PATH

echo "run on trainer:" ${PADDLE_TRAINER_ID}

cd models/

python -u train.py \
    --src_vocab_fpath '../cluster_test_data_en_fr/thirdparty/vocab.wordpiece.en-fr' \
    --trg_vocab_fpath '../cluster_test_data_en_fr/thirdparty/vocab.wordpiece.en-fr' \
    --special_token '<s>' '<e>' '<unk>'  \
    --token_delimiter '\x01' \
    --train_file_pattern '../cluster_test_data_en_fr/train/train.wordpiece.en-fr.0' \
    --val_file_pattern '../cluster_test_data_en_fr/thirdparty/newstest2014.wordpiece.en-fr' \
    --use_token_batch True \
    --batch_size  3200 \
    --sort_type pool \
    --pool_size 200000 \
    --local False \
  --shuffle True \
  --shuffle_batch True \
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
  weight_sharing True  > ../${PADDLE_TRAINER_ID}.log  2>&1 &
