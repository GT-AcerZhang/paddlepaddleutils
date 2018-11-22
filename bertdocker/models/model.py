import numpy as np
import paddle.fluid as fluid
from transformer_model import encoder


class BertModel(object):
    def __init__(self,
                 emb_size=1024,
                 mask_id=0,
                 masked_prob=0.15,
                 n_layer=12,
                 n_head=1,
                 voc_size=None,
                 max_position_seq_len=None,
                 pad_sent_id=None):
        self.emb_size = emb_size
        self.mask_id = mask_id
        self.masked_prob = masked_prob
        self.n_layer = n_layer
        self.n_head = n_head
        self.voc_size = voc_size
        self.max_position_seq_len = max_position_seq_len
        self.pad_sent_id = pad_sent_id

    def build_model(self, src_ids, position_ids, sentence_ids, self_attn_mask):
        emb_out = fluid.layers.embedding(
            input=src_ids,
            size=[self.voc_size + 1, self.emb_size],
            padding_idx=self.voc_size,
            is_sparse=True)

        position_emb_out = fluid.layers.embedding(
            input=position_ids,
            size=[self.max_position_seq_len + 1, self.emb_size],
            padding_idx=self.max_position_seq_len)

        sent_emb_out = fluid.layers.embedding(
            sentence_ids,
            size=[self.pad_sent_id + 1, self.emb_size],
            padding_idx=self.pad_sent_id)

        emb_out = emb_out + position_emb_out
        emb_out = emb_out + sent_emb_out

        enc_out = encoder(
            enc_input=emb_out,
            attn_bias=self_attn_mask,
            n_layer=self.n_layer,
            n_head=self.n_head,
            d_key=self.emb_size // self.n_head,
            d_value=self.emb_size // self.n_head,
            d_model=self.emb_size,
            d_inner_hid=self.emb_size * 4,
            prepostprocess_dropout=0.3,
            attention_dropout=0.1,
            relu_dropout=0.1,
            preprocess_cmd="n",
            postprocess_cmd="da")
        return enc_out

    def get_pooled_output(self, enc_out, next_sent_index):
        """Get the first feature of each sequence for classification"""
        reshaped_emb_out = fluid.layers.reshape(
            x=enc_out, shape=[-1, self.emb_size], inplace=True)
        next_sent_index = fluid.layers.cast(x=next_sent_index, dtype='int32')
        next_sent_feat = fluid.layers.gather(
            input=reshaped_emb_out, index=next_sent_index)
        return next_sent_feat

    def get_pretraining_output(self, enc_out, mask_label, mask_pos, labels,
                               next_sent_index):
        """Get the loss & accuracy for pretraining"""
        mask_pos = fluid.layers.cast(x=mask_pos, dtype='int32')

        # need reshape emb_out before gather
        reshaped_emb_out = fluid.layers.reshape(
            x=enc_out, shape=[-1, self.emb_size], inplace=True)
        mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=mask_pos)

        fc_out = fluid.layers.fc(input=mask_feat, size=self.voc_size)
        mask_lm_loss = fluid.layers.softmax_with_cross_entropy(
            logits=fc_out, label=mask_label)
        mean_mask_lm_loss = fluid.layers.reduce_mean(mask_lm_loss)

        #extract the first vector in each sentence
        next_sent_index = fluid.layers.cast(x=next_sent_index, dtype='int32')
        next_sent_feat = fluid.layers.gather(
            input=reshaped_emb_out, index=next_sent_index)
        next_sent_feat = fluid.layers.relu(next_sent_feat)
        next_sent_fc_out = fluid.layers.fc(input=next_sent_feat, size=2)

        next_sent_loss, next_sent_softmax = fluid.layers.softmax_with_cross_entropy(
            logits=next_sent_fc_out, label=labels, return_softmax=True)

        next_sent_acc = fluid.layers.accuracy(
            input=next_sent_softmax, label=labels)

        mean_next_sent_loss = fluid.layers.reduce_mean(next_sent_loss)

        loss = mean_next_sent_loss + mean_mask_lm_loss
        return next_sent_acc, mean_mask_lm_loss, loss

