#!/usr/bin/python
import sys
import os

if __name__ == '__main__':
    #cmd="python -u train.py --src_vocab_fpath thirdparty/vocab.wordpiece.en-fr --trg_vocab_fpath thirdparty/vocab.wordpiece.en-fr  --special_token '<s>' '<e>' '<unk>' --token_delimiter '\\x01' --train_file_pattern train_data/train.wordpiece.en-fr.'*' --val_file_pattern thirdparty/newstest2014.wordpiece.en-fr --use_token_batch True --batch_size  3200 --sort_type pool --pool_size 200000"
    #cmd = "python -u train.py --src_vocab_fpath thirdparty/vocab.bpe.32000 --trg_vocab_fpath thirdparty/vocab.bpe.32000 --special_token '<s>' '<e>' '<unk>' --train_file_pattern thirdparty/train.tok.clean.bpe.32000.en-de --token_delimiter ' ' --use_token_batch True --batch_size 2048 --sort_type pool --pool_size 200000"
    cmd = """
python -u train.py \
  --src_vocab_fpath test_data/vocab.bpe.32000 \
  --trg_vocab_fpath test_data/vocab.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --train_file_pattern test_data/train.tok.clean.bpe.32000.en-de.train_\* \
  --val_file_pattern test_data/newstest2013.tok.bpe.32000.en-de \
  --token_delimiter ' ' \
  --use_token_batch True \
  --batch_size 1024 \
  --sort_type pool \
  --pool_size 2000
"""
    print "cmd:", cmd
    exit(os.system(cmd))
