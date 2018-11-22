#!/usr/bin/env python
# -*- coding:utf8 -*-

############################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
############################################################
"""
Brief: Data IO for PyReader, For reference of PyReader, Please visit:
http://staging.paddlepaddle.org/documentation/docs/zh/1.0/user_guides/howto/prepare_data/use_py_reader.html

Author: tianxin(tianxin04@baidu.com)
Date: 2018/10/29 11:11:45
"""

from __future__ import print_function
from __future__ import division

import os
import numpy as np
import types
import gzip
import logging

import paddle
import paddle.fluid as fluid

from prepare_data import prepare_batch_data


class DataReader:
    def __init__(self,
                 data_dir,
                 place=None,
                 batch_size=4096,
                 max_seq_len=512,
                 num_head=1,
                 cls_id=1,
                 sep_id=2,
                 shuffle_files=True,
                 epoch=100,
                 voc_size=0,
                 pad_word_id=0,
                 pad_sent_id=3,
                 mask_id=0,
                 is_test=False):
                
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.place = place
        self.shuffle_files = shuffle_files
        self.epoch = epoch
        self.current_epoch = 0
        self.current_file_index = 0
        self.total_file = 0
        self.current_file = None
        self.voc_size = voc_size
        self.pad_word_id = pad_word_id
        self.pad_sent_id = pad_sent_id
        self.mask_id = mask_id
        self.max_seq_len = max_seq_len
        self.num_head = num_head
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.is_test = is_test
        assert self.batch_size > 100, "Current batch size means total token's number, \
                                       it should not be set to too small number."
        if self.is_test:
            self.epoch = 1
            self.shuffle_files = False

    def get_progress(self):
        """return current progress of traning data
        """
        return self.current_epoch, self.current_file_index, self.total_file, self.current_file

    def parse_line(self, line, max_seq_len=512):
        """ parse one line to token_ids, sentence_ids, pos_ids, label
        """
        line = line.strip().split(";")
        assert len(line) == 4, "One sample must have 4 fields!"
        (token_ids, sent_ids, pos_ids, label) = line
        token_ids = [int(token) for token in token_ids.split(" ")]
        sent_ids  = [int(token) for token in sent_ids.split(" ")]
        pos_ids   = [int(token) for token in pos_ids.split(" ")]
        assert len(token_ids) == len(sent_ids) == len(pos_ids), "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids)"
        label = int(label)
        if len(token_ids) > max_seq_len:
            return None
        return [token_ids, sent_ids, pos_ids, label]

    def data_generator(self):
        """
        data_generator
        """
        files = os.listdir(self.data_dir)
        self.total_file = len(files)
        def wrapper():
            def reader():
                for epoch in range(self.epoch):
                    self.current_epoch = epoch + 1
                    if self.shuffle_files:
                        np.random.shuffle(files)
                    for index, file in enumerate(files):
                        self.current_file_index = index+1
                        self.current_file = file
                        with gzip.open(self.data_dir + "/" + file, "rb") as f:
                        # with open(self.data_dir + "/" + file, "rb") as f:
                            for line in f:
                                parsed_line = self.parse_line(line, max_seq_len=self.max_seq_len)
                                if parsed_line is None:
                                    continue
                                else:
                                    yield parsed_line

            def batch_reader(reader, batch_size):
                batch, total_token_num, max_len = [], 0, 0
                for parsed_line in reader():
                    token_ids, sent_ids, pos_ids, label = parsed_line
                    max_len = max(max_len, len(token_ids))
                    if (len(batch) + 1) * max_len <= batch_size:
                        batch.append(parsed_line)
                        total_token_num += len(token_ids)
                    else:
                        yield batch, total_token_num
                        batch, total_token_num, max_len = [], 0, 0

                if len(batch) > 0:
                    yield batch, total_token_num

            #batch_reader = paddle.batch(reader, self.batch_size)
            for batch_data, total_token_num in batch_reader(reader, self.batch_size):
                # print("batch_data for prepare_batch_data")
                # print(batch_data)
                yield prepare_batch_data(
                    batch_data,
                    total_token_num,
                    n_head=self.num_head,
                    voc_size=self.voc_size,
                    pad_word_id=self.pad_word_id,
                    pad_sent_id=self.pad_sent_id,
                    pad_pos_id=self.max_seq_len,
                    mask_id=self.mask_id,
                    return_attn_bias=True,
                    return_max_len=False,
                    return_num_token=False,
                    place=self.place)

        return wrapper


if __name__ == "__main__":
    baike_bert_dataset_v1 = "/ssd2/liyukun01/bert/data/baike-bert-dataset-v1/"
    paddle_bert_dataset_v1 = "/ssd2/liyukun01/bert/data/paddle-bert-dataset-v1/"
    reader = DataReader(
        baike_bert_dataset_v1,
        paddle.fluid.CPUPlace(),
        batch_size=200,
        max_seq_len=50,
        shuffle_files=True,
        pad_word_id=300001,
        pad_sent_id=3,
        mask_id=0).data_generator()

    for batch in reader():
        (src_id, pos_id, sent_id, self_attn_mask, mask_label, mask_pos, labels, next_sent_index) = batch
        print("src_id data:{0}".format(src_id.shape))
        print("pos_id data:{0}".format(pos_id.shape))
        print("sent_id data:{0}".format(sent_id.shape))
        print("self_atten_mask shape:{0}".format(self_attn_mask.shape))
        print("mask_label shape:{0}".format(mask_label.shape))
        print("mask_pos shape:{0}".format(mask_pos.shape))
        print("labels shape:{0}".format(labels.shape))
        print("next_sent_index shape:{0}".format(next_sent_index.shape))
        #print("src_id data:{0}".format(src_id))
        #print("pos_id data:{0}".format(pos_id))
        #print("sent_id data:{0}".format(sent_id))
        #print("self_atten_mask shape:{0}".format(self_attn_mask))
        #print("mask_label shape:{0}".format(mask_label))
        #print("mask_pos shape:{0}".format(mask_pos))
        #print("labels shape:{0}".format(labels))
        #print("next_sent_index data:{0}".format(next_sent_index))
