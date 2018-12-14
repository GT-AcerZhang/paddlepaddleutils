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
import re
import tokenization

import paddle
import paddle.fluid as fluid

from prepare_data import prepare_batch_data

char_pattern = re.compile(ur'([,.!?\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b]|[\u4e00-\u9fa5]|[a-zA-Z0-9]+)')


class DataReader(object):
    def __init__(self,
                 data_dir,
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
                 is_test=False,
                 generate_neg_sample=False):

        self.data_dir = data_dir
        self.batch_size = batch_size
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
        self.generate_neg_sample = generate_neg_sample
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
        sent_ids = [int(token) for token in sent_ids.split(" ")]
        pos_ids = [int(token) for token in pos_ids.split(" ")]
        assert len(token_ids) == len(sent_ids) == len(
            pos_ids
        ), "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids)"
        label = int(label)
        if len(token_ids) > max_seq_len:
            return None
        return [token_ids, sent_ids, pos_ids, label]

    def read_file(self, file):
        assert file.endswith('.gz'), "[ERROR] %s is not a gzip file" % file
        file_path = self.data_dir + "/" + file 
        with gzip.open(file_path, "rb") as f:
            for line in f:
                parsed_line = self.parse_line(line, max_seq_len=self.max_seq_len)
                if parsed_line is None:
                    continue
                yield parsed_line

    def random_pair_neg_samples(self, pos_samples):
        """ randomly generate negtive samples using pos_samples

            Args:
                pos_samples: list of positive samples
            
            Returns:
                neg_samples: list of negtive samples
        """
        np.random.shuffle(pos_samples)
        num_sample = len(pos_samples)
        neg_samples = []
        miss_num = 0

        for i in range(num_sample):
            pair_index = (i + 1) % num_sample
            origin_src_ids = pos_samples[i][0]
            origin_sep_index = origin_src_ids.index(2)
            pair_src_ids = pos_samples[pair_index][0]
            pair_sep_index = pair_src_ids.index(2)

            src_ids = origin_src_ids[: origin_sep_index + 1] + pair_src_ids[pair_sep_index + 1 :]
            if len(src_ids) >= self.max_seq_len:
                miss_num += 1
                continue
            sent_ids = [1] * len(origin_src_ids[: origin_sep_index + 1]) + [2] * len(pair_src_ids[pair_sep_index + 1 :])
            pos_ids = list(range(len(src_ids)))
            neg_sample = [src_ids, sent_ids, pos_ids, 0]
            assert len(src_ids) == len(sent_ids) == len(pos_ids), "[ERROR]len(src_id) == lne(sent_id) == len(pos_id) must be True"
            neg_samples.append(neg_sample)
        return neg_samples, miss_num

    def mixin_negtive_samples(self, pos_sample_generator, buffer=1000):
        """ 1. generate negtive samples by randomly group sentence_1 and sentence_2 of positive samples
            2. combine negtive samples and positive samples
            
            Args:
                pos_sample_generator: a generator producing a parsed positive sample, which is a list: [token_ids, sent_ids, pos_ids, 1]

            Returns:
                sample: one sample from shuffled positive samples and negtive samples
        """
        pos_samples = []
        num_total_miss = 0
        pos_sample_num = 0
        try:
            while True:
                while len(pos_samples) < buffer:
                    pos_sample = next(pos_sample_generator)
                    label = pos_sample[3]
                    assert label == 1, "positive sample's label must be 1"
                    pos_samples.append(pos_sample)
                    pos_sample_num += 1

                #print("pos_samples:")
                #print(pos_samples)
                neg_samples, miss_num = self.random_pair_neg_samples(pos_samples)
                num_total_miss += miss_num
                #print("neg_samples:")
                #print(neg_samples)
                samples = pos_samples + neg_samples
                pos_samples = []
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
        except StopIteration:
            print("stopiteration: reach end of file")
            #print("len:%d" % len(pos_samples))
            if len(pos_samples) == 1:
                #print("pos_samples:")
                #print(pos_samples[0])
                yield pos_samples[0]
            elif len(pos_samples) == 0:
                yield None
            else:
                #print("pos_samples:")
                #print(pos_samples)
                neg_samples, miss_num = self.random_pair_neg_samples(pos_samples)
                num_total_miss += miss_num
                #print("neg_samples:")
                #print(neg_samples)
                samples = pos_samples + neg_samples
                pos_samples = []
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
            print("miss_num:%d\tideal_total_sample_num:%d\tmiss_rate:%f" % 
                 (num_total_miss, pos_sample_num * 2, num_total_miss / (pos_sample_num * 2)))

    def data_generator(self):
        """
        data_generator
        """
        files = os.listdir(self.data_dir)
        self.total_file = len(files)
        assert self.total_file > 0, "[Error] data_dir is empty"

        def wrapper():
            def reader():
                for epoch in range(self.epoch):
                    self.current_epoch = epoch + 1
                    if self.shuffle_files:
                        np.random.shuffle(files)
                    for index, file in enumerate(files):
                        self.current_file_index = index + 1
                        self.current_file = file
                        sample_generator =  self.read_file(file)
                        if not self.is_test and self.generate_neg_sample:
                            sample_generator = self.mixin_negtive_samples(sample_generator)
                        for sample in sample_generator:
                            if sample is None:
                                continue
                            yield sample

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
                        batch, total_token_num, max_len = [parsed_line], len(
                            token_ids), len(token_ids)

                if len(batch) > 0:
                    yield batch, total_token_num

            #batch_reader = paddle.batch(reader, self.batch_size)
            for batch_data, total_token_num in batch_reader(reader,
                                                            self.batch_size):
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
                    return_num_token=False)

        return wrapper


class XnliDataReader(DataReader):
    def __init__(self,
                 data_dir,
                 vocab_path,
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
                 unk_id=9999,
                 char_based=True):
        super(XnliDataReader, self).__init__(
            data_dir, batch_size, max_seq_len, num_head, cls_id, sep_id,
            shuffle_files, epoch, voc_size, pad_word_id, pad_sent_id, mask_id)

        self.vocab_path = vocab_path
        self.char_based = char_based
        self.vocab = {}
        self.label_dict = {
            "entailment": 0,
            "contradictory": 1,
            "contradiction": 1,
            "neutral": 2
        }
        self.unk_id = unk_id

        with open(self.vocab_path, "rb") as vocab_file:
            line = vocab_file.readline()
            while line:
                line = line.strip().split("\t")
                self.vocab[line[0].decode('gb18030')] = int(line[1])
                line = vocab_file.readline()

    def parse_line(self, line, max_seq_len=512):
        if self.char_based:
            line = line.decode('utf8').replace(' ', '').split("\t")
        else:
            pass
        assert len(line) == 3, "One XNLI training sample must have 3 fields!"

        (first_sent, second_sent, label) = line

        first_sent_ids = [
            self.vocab.get(token, self.unk_id) \
                    for token in char_pattern.findall(first_sent)
        ]
        second_sent_ids = [
            self.vocab.get(token, self.unk_id) \
                    for token in char_pattern.findall(second_sent)
        ]

        token_ids = [self.cls_id] + first_sent_ids + [
            self.sep_id
        ] + second_sent_ids + [self.sep_id]
        sent_ids = [1] * (len(first_sent_ids) + 2) + [2] * (
            len(second_sent_ids) + 1)
        pos_ids = list(range(len(sent_ids)))
        assert len(token_ids) == len(sent_ids) == len(
            pos_ids
        ), "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids)"
        label_id = self.label_dict.get(label.strip(), -1)
        if len(token_ids) > max_seq_len or label_id == -1:
            print("label err", label)
            return None
        return [token_ids, sent_ids, pos_ids, label_id]

    def data_generator(self, lang="zh", phase="train"):
        def train_reader():
            files = ["multinli/multinli.train." + lang + ".tsv"]
            self.total_file = len(files)
            for epoch in range(self.epoch):
                self.current_epoch = epoch + 1
                if self.shuffle_files:
                    np.random.shuffle(files)
                for index, file in enumerate(files):
                    self.current_file_index = index + 1
                    self.current_file = file
                    with open(os.path.join(self.data_dir, file), "rb") as f:
                        lines = f.readlines()[1:] #skip first line of xnli trainset
                        np.random.shuffle(lines)
                        for line in lines:
                            parsed_line = self.parse_line(
                                line, max_seq_len=self.max_seq_len)
                            if parsed_line is None:
                                continue
                            else:
                                yield parsed_line

        def test_and_dev_reader():
            files = ["xnli." + phase + ".tsv"]
            for index, file in enumerate(files):
                with open(os.path.join(self.data_dir, file), "rb") as f:
                    # with open(self.data_dir + "/" + file, "rb") as f:
                    line = f.readline()
                    while line:
                        items = line.strip().split("\t")
                        line = f.readline()
                        if items[0].decode('utf8') != lang:
                            continue
                        norm_line = items[6] + "\t" + items[7] + "\t" + items[
                            1]
                        parsed_line = self.parse_line(
                            norm_line, max_seq_len=self.max_seq_len)
                        if parsed_line is None:
                            continue
                        else:
                            yield parsed_line

        def wrapper():
            if phase == "train":
                reader = train_reader
            else:
                reader = test_and_dev_reader

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
                        batch, total_token_num, max_len = [parsed_line], len(
                            token_ids), len(token_ids)

                if len(batch) > 0:
                    yield batch, total_token_num

            for batch_data, total_token_num in batch_reader(reader,
                                                            self.batch_size):
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
                    return_num_token=False)

        return wrapper

class XnliDataReaderOfficial(DataReader):
    def __init__(self,
                 data_dir,
                 vocab_path,
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
                 char_based=True):
        super(XnliDataReaderOfficial, self).__init__(
            data_dir, batch_size, max_seq_len, num_head, cls_id, sep_id,
            shuffle_files, epoch, voc_size, pad_word_id, pad_sent_id, mask_id)

        self.vocab_path = vocab_path
        self.char_based = char_based
        self.label_dict = {
            "entailment": 0,
            "contradictory": 1,
            "contradiction": 1,
            "neutral": 2
        }
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=True)
        
    def parse_line(self, line, max_seq_len=512):
        if self.char_based:
            line = line.decode('utf8').replace(' ', '').split("\t")
        else:
            pass
        assert len(line) == 3, "One XNLI training sample must have 3 fields!"

        (first_sent, second_sent, label) = line

        first_sent_ids =self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(first_sent))
        second_sent_ids =self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(second_sent))

        token_ids = [self.cls_id] + first_sent_ids + [
            self.sep_id
        ] + second_sent_ids + [self.sep_id]
        sent_ids = [1] * (len(first_sent_ids) + 2) + [2] * (
            len(second_sent_ids) + 1)
        pos_ids = list(range(len(sent_ids)))
        assert len(token_ids) == len(sent_ids) == len(
            pos_ids
        ), "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids)"
        label_id = self.label_dict.get(label.strip(), -1)
        if len(token_ids) > max_seq_len or label_id == -1:
            print("label err", label)
            return None
        return [token_ids, sent_ids, pos_ids, label_id]

    def data_generator(self, lang="zh", phase="train"):
        def train_reader():
            files = ["multinli/multinli.train." + lang + ".tsv"]
            self.total_file = len(files)
            for epoch in range(self.epoch):
                self.current_epoch = epoch + 1
                if self.shuffle_files:
                    np.random.shuffle(files)
                for index, file in enumerate(files):
                    self.current_file_index = index + 1
                    self.current_file = file
                    with open(os.path.join(self.data_dir, file), "rb") as f:
                        lines = f.readlines()[1:] #skip first line of xnli trainset
                        np.random.shuffle(lines)
                        for line in lines:
                            parsed_line = self.parse_line(
                                line, max_seq_len=self.max_seq_len)
                            if parsed_line is None:
                                continue
                            else:
                                yield parsed_line

        def test_and_dev_reader():
            files = ["xnli." + phase + ".tsv"]
            for index, file in enumerate(files):
                with open(os.path.join(self.data_dir, file), "rb") as f:
                    # with open(self.data_dir + "/" + file, "rb") as f:
                    line = f.readline()
                    while line:
                        items = line.strip().split("\t")
                        line = f.readline()
                        if items[0].decode('utf8') != lang:
                            continue
                        norm_line = items[6] + "\t" + items[7] + "\t" + items[
                            1]
                        parsed_line = self.parse_line(
                            norm_line, max_seq_len=self.max_seq_len)
                        if parsed_line is None:
                            continue
                        else:
                            yield parsed_line

        def wrapper():
            if phase == "train":
                reader = train_reader
            else:
                reader = test_and_dev_reader

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
                        batch, total_token_num, max_len = [parsed_line], len(
                            token_ids), len(token_ids)

                if len(batch) > 0:
                    yield batch, total_token_num

            for batch_data, total_token_num in batch_reader(reader,
                                                            self.batch_size):
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
                    return_num_token=False)

        return wrapper

if __name__ == "__main__":
    baike_bert_dataset_v1 = "/ssd2/liyukun01/bert/data/baike-bert-dataset-v1/"
    paddle_bert_dataset_v1 = "/ssd2/liyukun01/bert/data/paddle-bert-dataset-v1/"
    pos_sample = "./data/pos_sample/"
    #pos_sample = "/ssd2/liyukun01/bert/data/baike-bert-dataset-char-v3_test/"
    reader = DataReader(
        pos_sample,
        paddle.fluid.CPUPlace(),
        voc_size=10005,
        batch_size=4096,
        max_seq_len=512,
        shuffle_files=True,
        pad_word_id=10005,
        pad_sent_id=3,
        epoch=1,
        is_test=False,
        mask_id=0,
        generate_neg_sample=True).data_generator()

    for batch in reader():
        (src_id, pos_id, sent_id, self_attn_mask, mask_label, mask_pos, labels,
         next_sent_index) = batch
        '''
        print("src_id data:{0}".format(src_id.shape))
        print("pos_id data:{0}".format(pos_id.shape))
        print("sent_id data:{0}".format(sent_id.shape))
        print("self_atten_mask shape:{0}".format(self_attn_mask.shape))
        print("mask_label shape:{0}".format(mask_label.shape))
        print("mask_pos shape:{0}".format(mask_pos.shape))
        print("labels shape:{0}".format(labels.shape))
        print("next_sent_index shape:{0}".format(next_sent_index.shape))
        '''
        #print("src_id data:{0}".format(src_id))
        #print("pos_id data:{0}".format(pos_id))
        #print("sent_id data:{0}".format(sent_id))
        #print("self_atten_mask shape:{0}".format(self_attn_mask))
        #print("mask_label shape:{0}".format(mask_label))
        #print("mask_pos shape:{0}".format(mask_pos))
        #print("labels shape:{0}".format(labels))
        #print("next_sent_index data:{0}".format(next_sent_index))
