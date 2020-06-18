export LD_LIBRARY_PATH=/root/go/soft/env/cuda-9.0/lib64:/root/go/soft/cuda10-cudnn7.6.5.32/lib64:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
export CUDA_VISIBLE_DEVICES=4
python3.6 ./text_classifier.py --checkpoint_dir ckpt_20200618030456
