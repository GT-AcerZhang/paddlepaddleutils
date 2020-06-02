# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np
import sys
import math
import argparse
from reader import ChnSentiCorp

CLASS_DIM = 2
EMB_DIM = 128
HID_DIM = 512
BATCH_SIZE = 128


def parse_args():
    parser = argparse.ArgumentParser("conv")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help="If set, run the task with continuous evaluation logs.")
    parser.add_argument(
        '--use_gpu', type=int, default=0, help="Whether to use GPU or not.")
    parser.add_argument(
        '--num_epochs', type=int, default=1, help="number of epochs.")
    args = parser.parse_args()
    return args


def convolution_net(data, input_dim, class_dim, emb_dim, hid_dim, test=False):
    emb = fluid.embedding(input=data, size=[input_dim, emb_dim], is_sparse=True)
    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=3,
        act="tanh",
        pool_type="sqrt")
    conv_4 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=4,
        act="tanh",
        pool_type="sqrt")
    if test:
        prediction = fluid.layers.fc(
            input=[conv_3, conv_4], size=class_dim, act='softmax')
    else:
        prediction = fluid.layers.fc(
            input=[conv_3, conv_4], size=class_dim)
    return prediction


def inference_program(word_dict):
    dict_dim = len(word_dict)
    data = fluid.data(name="words", shape=[None], dtype="int64", lod_level=1)
    net = convolution_net(data, dict_dim, CLASS_DIM, EMB_DIM, HID_DIM, test=True)
    return net


def train_program(prediction, teacher_predict_dim):
    label = fluid.data(name="label", shape=[None, 1], dtype="int64")

    teacher_logits = fluid.data(
        name='teacher_logits',
        shape=[None, HID_DIM],
        dtype='float32')

    cost = fluid.layers.softmax_with_cross_entropy(prediction, label=teacher_logits)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)
    return [avg_cost, accuracy]


def optimizer_func():
    return fluid.optimizer.Adagrad(learning_rate=0.002)


def get_conn_data(batch_data):
    s = []
    t = []
    for  r in batch_data:
        s.append((r[0], r[1]))
        t.append(r[2])

    return s, r

def train(use_cuda, params_dirname):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    ds = ChnSentiCorp()
    word_dict = ds.student_word_dict("./data/vocab.bow.txt")

    t_reader = ds.student_reader("./data/train.part.0", word_dict)
    d_reader = ds.student_reader("./data/dev.part.0", word_dict)

    if args.enable_ce:
        train_reader = paddle.batch(t_reader, batch_size=BATCH_SIZE)
    else:
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                t_reader, buf_size=25000),
            batch_size=BATCH_SIZE)

    test_reader = paddle.batch(
        d_reader, batch_size=BATCH_SIZE)

    feed_order = ['words', 'label']
    pass_num = args.num_epochs

    main_program = fluid.default_main_program()
    star_program = fluid.default_startup_program()

    if args.enable_ce:
        main_program.random_seed = 90
        star_program.random_seed = 90

    prediction = inference_program(word_dict)
    train_func_outputs = train_program(prediction)
    avg_cost = train_func_outputs[0]

    test_program = main_program.clone(for_test=True)

    # [avg_cost, accuracy] = train_program(prediction)
    sgd_optimizer = optimizer_func()
    sgd_optimizer.minimize(avg_cost)
    exe = fluid.Executor(place)

    def train_test(program, reader):
        count = 0
        feed_var_list = [
            program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder_test = fluid.DataFeeder(feed_list=feed_var_list, place=place)
        test_exe = fluid.Executor(place)
        accumulated = len(train_func_outputs) * [0]
        for r in reader():
            test_data, _ = get_conn_data(r)
            avg_cost_np = test_exe.run(
                program=program,
                feed=feeder_test.feed(test_data),
                fetch_list=train_func_outputs)
            accumulated = [
                x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)
            ]
            count += 1
        return [x / count for x in accumulated]

    def train_loop():

        feed_var_list_loop = [
            main_program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
        exe.run(star_program)

        for epoch_id in range(pass_num):
            for step_id, r in enumerate(train_reader()):
                student_data, teacher_data = get_conn_data(r)
                data = student_data
                metrics = exe.run(
                    main_program,
                    feed=feeder.feed(data),
                    fetch_list=[var.name for var in train_func_outputs])
                print("step: {0}, Metrics {1}".format(
                    step_id, list(map(np.array, metrics))))
                if (step_id + 1) % 10 == 0:
                    avg_cost_test, acc_test = train_test(test_program,
                                                         test_reader)
                    print('Step {0}, Test Loss {1:0.2}, Acc {2:0.2}'.format(
                        step_id, avg_cost_test, acc_test))

                    print("Step {0}, Epoch {1} Metrics {2}".format(
                        step_id, epoch_id, list(map(np.array, metrics))))
                if math.isnan(float(metrics[0])):
                    sys.exit("got NaN loss, training failed.")
            if params_dirname is not None:
                fluid.io.save_inference_model(params_dirname, ["words"],
                                              prediction, exe)
            if args.enable_ce and epoch_id == pass_num - 1:
                print("kpis\tconv_train_cost\t%f" % metrics[0])
                print("kpis\tconv_train_acc\t%f" % metrics[1])
                print("kpis\tconv_test_cost\t%f" % avg_cost_test)
                print("kpis\tconv_test_acc\t%f" % acc_test)

    train_loop()

def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    params_dirname = "understand_sentiment_conv.inference.model"
    train(use_cuda, params_dirname)


if __name__ == '__main__':
    args = parse_args()
    use_cuda = args.use_gpu  # set to True if training with GPU
    main(use_cuda)
