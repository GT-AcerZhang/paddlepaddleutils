#!/usr/bin/env python
# -*- coding:utf8 -*-

############################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
############################################################
"""
Brief: predict module for bert.
    Key Obejectives:
    1. predict test set using existed model independent of training progress.
    2. predict validation set when training progress going on.

Author: tianxin(tianxin04@baidu.com)
Date: 2018/11/12 17:33:18
"""

import time
import numpy as np

import paddle.fluid as fluid
from model import BertModel
from reader_tx import DataReader
from utils import init_model


def predict_wrapper(args,
                    test_prog=None,
                    train_exe=None,
                    pyreader=None,
                    fetch_list=None):
    # Context to do validation.
    data_path = args.test_set_dir if args.for_test else args.validation_set_dir
    data_reader = DataReader(
        data_path,
        batch_size=args.batch_size,
        voc_size=args.vocab_size,
        pad_word_id=args.vocab_size,
        pad_sent_id=3,
        shuffle_files=False,
        epoch=1,
        max_seq_len=args.max_seq_len,
        is_test=True,
        num_head=args.num_head)

    if args.for_test:
        init_model(args.init_model, test_prog)
        test_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda, main_program=test_prog)
    else:
        test_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            main_program=test_prog,
            share_vars_from=train_exe)

    def predict(exe=test_exe, pyreader=pyreader):

        pyreader.decorate_tensor_provider(data_reader.data_generator())
        pyreader.start()

        cost = 0
        lm_cost = 0
        acc = 0
        steps = 0
        time_begin = time.time()
        while True:
            try:
                each_next_acc, each_mask_lm_cost, each_total_cost = exe.run(
                    fetch_list=fetch_list)
                acc += each_next_acc
                lm_cost += each_mask_lm_cost
                cost += each_total_cost
                steps += 1
                if args.for_test and steps % args.skip_steps == 0:
                    print("[test_set] steps: %d" % steps)

            except fluid.core.EOFException:
                pyreader.reset()
                break

        used_time = time.time() - time_begin
        return cost, lm_cost, acc, steps, (args.skip_steps / used_time)

    return predict
