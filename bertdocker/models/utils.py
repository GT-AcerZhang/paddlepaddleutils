#!/usr/bin/env python
# -*- coding:utf8 -*-

############################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
############################################################
"""
Brief: utils for bert

Author: tianxin(tianxin04@baidu.com)
Date: 2018/10/29 11:11:45
"""

import os
import six
import argparse
import ast
import copy

import numpy as np
import paddle.fluid as fluid


def init_model(init_model_path, main_program):
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    assert os.path.exists(
        init_model_path), "[%s] cann't be found." % init_model_path
    fluid.io.load_persistables(exe, init_model_path, main_program=main_program)
    print("Load model from {}".format(init_model_path))


def init_pretraining_model(exe, pretraining_model_path, main_program):
    assert os.path.exists(pretraining_model_path
                          ), "[%s] cann't be found." % pretraining_model_path

    def existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(os.path.join(pretraining_model_path, var.name))

    fluid.io.load_vars(
        exe,
        pretraining_model_path,
        main_program=main_program,
        predicate=existed_params)
    print("Load pretraining model from {}".format(pretraining_model_path))


def append_nccl2_prepare(startup_prog, trainer_id, worker_endpoints,
                         current_endpoint):
    assert (trainer_id >= 0 and len(worker_endpoints) > 1 and
            current_endpoint in worker_endpoints)
    eps = copy.deepcopy(worker_endpoints)
    eps.remove(current_endpoint)
    nccl_id_var = startup_prog.global_block().create_var(
        name="NCCLID", persistable=True, type=fluid.core.VarDesc.VarType.RAW)
    startup_prog.global_block().append_op(
        type="gen_nccl_id",
        inputs={},
        outputs={"NCCLID": nccl_id_var},
        attrs={
            "endpoint": current_endpoint,
            "endpoint_list": eps,
            "trainer_id": trainer_id
        })
    return nccl_id_var


def parse_args():
    parser = argparse.ArgumentParser("BERT training.")
    parser.add_argument(
        '--epoch',
        type=int,
        default=100,
        help='epoch for training. (default: %(default)d)')
    parser.add_argument(
        '--d_model',
        type=int,
        default=1024,
        help='Batch size for training. (default: %(default)d)')
    parser.add_argument(
        '--num_head',
        type=int,
        default=1,
        help='Attention head. (default: %(default)d)')
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=512,
        help='Number of word of the longest seqence. (default: %(default)d)')
    parser.add_argument(
        '--num_layers',
        type=int,
        default=12,
        help='Batch size for training. (default: %(default)d)')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8096,
        help='Total token number in batch for training. (default: %(default)d)')
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=300005,
        help='vocab_szie. (default: %(default)d)')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1.0,
        help='Learning rate used to train with warmup. (default: %(default)f)')
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=4000,
        help='warmup steps. (default: %(default)f)')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help='Weight decay rate for L2 regularizer. (default: %(default)f)')
    parser.add_argument(
        '--checkpoints',
        type=str,
        default="checkpoints",
        help='Path to save checkpoints. (default: %(default)s)')
    parser.add_argument(
        '--init_model',
        type=str,
        default=None,
        help='init model to load. (default: %(default)s)')
    parser.add_argument(
        '--vocab_path',
        type=str,
        default=None,
        help='Vocabulary path. (default: %(default)s)')
    parser.add_argument(
        '--data_dir',
        type=str,
        default="./real_data",
        help='Path of training data. (default: %(default)s)')
    parser.add_argument(
        '--validation_set_dir',
        type=str,
        default="",
        help='Path of validation set. (default: %(default)s)')
    parser.add_argument(
        '--test_set_dir',
        type=str,
        default="",
        help='Path of test set. (default: %(default)s)')
    parser.add_argument(
        '--skip_steps',
        type=int,
        default=10,
        help='The steps interval to print loss. (default: %(default)d)')
    parser.add_argument(
        '--save_steps',
        type=int,
        default=10000,
        help='The steps interval to save checkpoints. (default: %(default)d)')
    parser.add_argument(
        '--pad_sent_id',
        type=int,
        default=3,
        help='pad_sent_id=3, %(default)d)')
    parser.add_argument(
        '--validation_steps',
        type=int,
        default=1000,
        help='The steps interval to evaluate model performance on validation '
        'set. (default: %(default)d)')
    parser.add_argument(
        '--is_distributed',
        action='store_true',
        help='If set, then start distributed training')
    parser.add_argument(
        '--use_cuda',
        action='store_true',
        help='If set, use GPU for training.')
    parser.add_argument(
        '--for_test',
        action='store_true',
        help='If set, evaluate existed model performance on test set.')
    parser.add_argument(
        '--generate_neg_sample',
        action='store_true',
        help='If set, randomly generate negtive samples by positive samples.')
    parser.add_argument(
        '--weight_sharing',
        action='store_true',
        help='If set, share weights between word embedding and masked lm '
        'output fully-connected layer.')
    parser.add_argument(
        '--use_fast_executor',
        action='store_true',
        help='If set, use fast parallel executor (in experiment).')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='If set profile it.')


    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def to_lodtensor(data, place):
    """
    convert to LODtensor
    """
    seq_lens = [[len(seq) for seq in data]]
    #print("total seq len: %d, padded len: %d"
    #        % (np.sum(seq_lens[0]), max(seq_lens[0]) * len(seq_lens[0])))
    flatten_data = [x for sublist in data for x in sublist]
    lod_tensor = fluid.create_lod_tensor(
        np.array([flatten_data]).reshape([-1, 1]).astype("int64"), seq_lens,
        place)
    return [lod_tensor]
