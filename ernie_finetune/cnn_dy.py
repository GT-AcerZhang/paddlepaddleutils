import sys
import os

import numpy as np
import argparse
from sklearn.metrics import f1_score
import paddle as P
import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D
from reader import ChnSentiCorp
from paddle_edl.distill.distill_reader import DistillReader
import re

import os
import sys
from paddle_serving_client import Client
from paddle_serving_app.reader import ChineseBertReader

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--fixed_teacher", type=str, default=None, help="fixed teacher for debug local distill")
args = parser.parse_args()

EPOCH=10
LR=5e-5

class AdamW(F.optimizer.AdamOptimizer):
    """AdamW object for dygraph"""
    def __init__(self, *args, **kwargs):
        weight_decay = kwargs.pop('weight_decay', None)
        var_name_to_exclude = kwargs.pop('var_name_to_exclude', '.*layer_norm_scale|.*layer_norm_bias|.*b_0')
        super(AdamW, self).__init__(*args, **kwargs)
        self.wd = weight_decay
        self.pat = re.compile(var_name_to_exclude)

    def apply_optimize(self, loss, startup_program, params_grads):
        super(AdamW, self).apply_optimize(loss, startup_program, params_grads)
        for p, g in params_grads:
            #log.debug(L.reduce_mean(p))
            if not self.pat.match(p.name):
                L.assign(p * (1. - self.wd * self.current_step_lr()), p)
            #log.debug(L.reduce_mean(p))

def KL(pred, target):
    pred = L.log(L.softmax(pred))
    target = L.softmax(target)
    loss = L.kldiv_loss(pred, target)
    return loss

def evaluate_student(model, dataset):
    all_pred, all_label = [], []
    with D.base._switch_tracer_mode_guard_(is_train=False):
        model.eval()
        for step, (ids_student, labels, ids) in enumerate(dataset.start()):
            _, logits = model(ids_student)
            pred = L.argmax(logits, -1)
            all_pred.extend(pred.numpy())
            all_label.extend(labels.numpy())
        f1 = f1_score(all_label, all_pred, average='macro')
        model.train()
        return f1 

class CNN(D.Layer):
    def __init__(self, word_dict):
        super().__init__()
        self.emb = D.Embedding([len(word_dict), 128], padding_idx=0)
        self.cnn = D.Conv2D(128, 128, (1, 3), padding=(0, 1), act='relu')
        self.pool = D.Pool2D((1, 3), pool_padding=(0, 1))
        self.fc = D.Linear(128, 2)
    def forward(self, ids, labels=None):
        embbed = self.emb(ids)
        #d_batch, d_seqlen = ids.shape
        hidden = embbed
        hidden = L.transpose(hidden, [0, 2, 1]) #change to NCWH
        hidden = L.unsqueeze(hidden, [2])
        hidden = self.cnn(hidden)
        hidden = self.pool(hidden)
        hidden = L.squeeze(hidden, [2])
        hidden = L.transpose(hidden, [0, 2, 1])
        pad_mask = L.unsqueeze(L.cast(ids!=0, 'float32'), [-1])
        hidden = L.softsign(L.reduce_sum(hidden * pad_mask, 1))
        logits = self.fc(hidden)
        if labels is not None:
            if len(labels.shape)==1:
                labels = L.reshape(labels, [-1, 1])
            loss = L.softmax_with_cross_entropy(logits, labels)
            loss = L.reduce_mean(loss)
        else:
            loss = None
        return loss, logits

def train(train_reader, test_reader, word_dict):
    model = CNN(word_dict)
    g_clip = F.clip.GradientClipByGlobalNorm(1.0) #experimental
    opt = AdamW(learning_rate=LR, parameter_list=model.parameters(), weight_decay=0.01, grad_clip=g_clip)
    model.train()
    for epoch in range(EPOCH):
        for step, (_,_,_,_,ids_student,label,logits_t) in enumerate(train_reader()):
            _, logits_s = model(ids_student) # student 模型输出logits
            loss_ce, _ = model(ids_student, labels=label)
            loss_kd = KL(logits_s, logits_t)    # 由KL divergence度量两个分布的距离
            loss = loss_ce + loss_kd
            loss.backward()
            if step % 10 == 0:
                print('[step %03d] distill train loss %.5f lr %.3e' % (step, loss.numpy(), opt.current_step_lr()))
            opt.minimize(loss)
            model.clear_gradients()
        f1 = evaluate_student(model, test_reader)
        print('student f1 %.5f' % f1)

def get_reader(train_reader, key_list):
    bert_reader = ChineseBertReader({'max_seq_len':128})
    def reader():
        for r in train_reader():
            feed_dict = bert_reader.process(r[2])
            #print(type(feed_dict))
            #sys.exit(0)
            l = []
            for k in key_list:
                l.append(feed_dict[k])
            l.appendr.extend([r[0], r[1]])
            yield l

    return reader

if __name__ == "__main__":
    place = F.CUDAPlace(0)
    D.guard(place).__enter__()

    ds = ChnSentiCorp()
    word_dict = ds.student_word_dict("./data/vocab.bow.txt")

    t_reader = ds.student_reader("./data/train.part.0", word_dict)
    d_reader = ds.student_reader("./data/dev.part.0", word_dict)

    feed_keys = ["input_ids", "position_ids", "segment_ids", "input_mask", "ids_student", "label"]
    train_reader = P.batch(
        P.reader.shuffle(
            get_reader(t_reader, feed_keys), buf_size=25000),
        batch_size=16)

    test_reader = P.batch(
        d_reader, batch_size=16)

    dr = DistillReader(feed_keys, predicts=['pooled_output'])
    dr.set_teacher_batch_size(2)
    dr.set_serving_conf_file("./ernie_senti_client/serving_client_conf.prototxt")
    if args.fixed_teacher:
        dr.set_fixed_teacher(args.fixed_teacher)
    train_reader = dr.set_sample_list_generator(train_reader)

    train(train_reader, test_reader, word_dict)
