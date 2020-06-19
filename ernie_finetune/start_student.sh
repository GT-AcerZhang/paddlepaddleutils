export LD_LIBRARY_PATH=/root/go/soft/env/cuda-9.0/lib64:/root/go/soft/cuda10-cudnn7.6.5.32/lib64:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
export CUDA_VISIBLE_DEVICES=3
for w in {1..50}
do
    echo "begin to calc s_weight:$w" 
    python3.6 -u cnn_dy.py --fixed_teacher 127.0.0.1:19294,127.0.0.1:19295,127.0.0.1:19296 > log/$w.log 2>&1
done
