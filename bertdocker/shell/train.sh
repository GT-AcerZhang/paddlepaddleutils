set -eu

#bash -x ./env.sh
source ./shell/env.sh
source ./shell/utils.sh

source ./model_conf

export PATH="$PWD/python/bin/:$PATH"
export PYTHONPATH="$PWD/python/"

export FLAGS_fraction_of_gpu_memory_to_use=1.0

#pack output
#nohup sh ./shell/pack_model.sh ./output > log/pack_model.log 2>&1 &

# check
check_iplist

mkdir -p log

python -u ./train.py --use_cuda \
                    --use_fast_executor \
                    --batch_size ${BATCH_SIZE} \
                    --data_dir ./data \
                    --validation_set_dir ${testdata_dir} \
                    --checkpoints ./output \
                    --save_steps ${SAVE_STEPS} \
                    --init_model ${init_model:-""} \
                    --learning_rate ${LR_RATE} \
                    --max_seq_len ${MAX_LEN} \
                    --vocab_size ${VOCAB_SIZE} \
                    --num_head ${NUM_HEAD} \
                    --d_model ${D_MODEL} \
                    --num_layers ${NUM_LAYER} \
                    --is_distributed \
                    --skip_steps 10 > log/job.log 2>&1
