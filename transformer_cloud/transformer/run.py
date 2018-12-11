#!/usr/bin/python
from __future__ import print_function
import sys
import os
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--profile', action='store_true')
parser.add_argument('--pserver_profile_end_step', type=int, default=-1)
parser.add_argument('--pserver_profile_path', type=str, default="./transformer_pserver")
parser.add_argument('--update_method', type=str, default="pserver")
parser.add_argument('--data_set', type=str, default="en-fr")
args = parser.parse_args()

def print_arguments():
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')

if __name__ == '__main__':
    if args.data_set == "en-fr":
        #en-fr without inference file
        cmd="FLAGS_fraction_of_gpu_memory_to_use=0.1 python -u train.py --src_vocab_fpath thirdparty/vocab.wordpiece.en-fr --trg_vocab_fpath thirdparty/vocab.wordpiece.en-fr  --special_token '<s>' '<e>' '<unk>' --token_delimiter '\\x01' --train_file_pattern train_data/train.wordpiece.en-fr.\* --use_token_batch True --batch_size  3200 --sort_type pool --pool_size 200000"
    elif args.data_set == "en-de":
        cmd = "FLAGS_fraction_of_gpu_memory_to_use=0.1 python -u train.py --src_vocab_fpath thirdparty/vocab.bpe.32000 --trg_vocab_fpath thirdparty/vocab.bpe.32000 --special_token '<s>' '<e>' '<unk>' --train_file_pattern thirdparty/train.tok.clean.bpe.32000.en-de --token_delimiter ' ' --use_token_batch True --batch_size 2048 --sort_type pool --pool_size 200000"
    else:
        print("not supported data_set:", args.data_set)

    if args.update_method == "nccl2":
        cmd = "NCCL_IB_DISABLE=1 " + cmd + " --update_method nccl2"

    if args.profile:
        assert args.pserver_profile_end_step >= 0
        assert len(args.pserver_profile_path) > 0
        cmd = "FLAGS_rpc_server_profile_period={} FLAGS_rpc_server_profile_path={} {}".format(args.pserver_profile_end_step, args.pserver_profile_path, cmd)

    print("cmd:", cmd)
    exit(os.system(cmd))
