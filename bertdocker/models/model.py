import numpy as np
import paddle.fluid as fluid
from transformer_model import encoder
from transformer_model import pre_process_layer

class BertModel(object):
    def __init__(
            self,
            src_ids,
            position_ids,
            sentence_ids,
            self_attn_mask,
            emb_size=1024,
            mask_id=0,
            masked_prob=0.15,
            n_layer=12,
            n_head=1,
            voc_size=10005,
            max_position_seq_len=512,
            pad_sent_id=3,
            weight_sharing=True, ):
        self._emb_size = emb_size
        self._mask_id = mask_id
        self._masked_prob = masked_prob
        self._n_layer = n_layer
        self._n_head = n_head
        self._voc_size = voc_size
        self._max_position_seq_len = max_position_seq_len
        self._pad_sent_id = pad_sent_id
        self._weight_sharing = weight_sharing

        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._sent_emb_name = "sent_embedding"

        self._build_model(src_ids, position_ids, sentence_ids, self_attn_mask)

    def _build_model(self, src_ids, position_ids, sentence_ids,
                     self_attn_mask):
        emb_out = fluid.layers.embedding(
            input=src_ids,
            size=[self._voc_size + 1, self._emb_size],
            padding_idx=self._voc_size,
            param_attr=fluid.ParamAttr(name=self._word_emb_name),
            is_sparse=False)

        position_emb_out = fluid.layers.embedding(
            input=position_ids,
            size=[self._max_position_seq_len + 1, self._emb_size],
            param_attr=fluid.ParamAttr(name=self._pos_emb_name),
            padding_idx=self._max_position_seq_len)

        sent_emb_out = fluid.layers.embedding(
            sentence_ids,
            size=[self._pad_sent_id + 1, self._emb_size],
            param_attr=fluid.ParamAttr(name=self._sent_emb_name),
            padding_idx=self._pad_sent_id)

        emb_out = emb_out + position_emb_out
        emb_out = emb_out + sent_emb_out
        
        #drop
        pre_process_dropout = 0.1
        emb_out = pre_process_layer(emb_out, 'nd',
                pre_process_dropout)

        n_head_self_attn_mask = fluid.layers.stack(x=[self_attn_mask]*self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient=True
        
        self._enc_out = encoder(
            enc_input=emb_out,
            attn_bias=n_head_self_attn_mask,
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            prepostprocess_dropout=0.1,
            attention_dropout=0.1,
            relu_dropout=0.1,
            preprocess_cmd="",
            postprocess_cmd="dan")

    def get_pooled_output(self, next_sent_index):
        """Get the first feature of each sequence for classification"""
        self._reshaped_emb_out = fluid.layers.reshape(
            x=self._enc_out, shape=[-1, self._emb_size], inplace=True)
        next_sent_index = fluid.layers.cast(x=next_sent_index, dtype='int32')
        next_sent_feat = fluid.layers.gather(
            input=self._reshaped_emb_out, index=next_sent_index)
        next_sent_feat = fluid.layers.fc(
            input=next_sent_feat,
            size=self._emb_size,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="pooled_fc_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr="pooled_fc_b")
        return next_sent_feat

    def get_pretraining_output(self, mask_label, mask_pos, labels,
                               next_sent_index):
        """Get the loss & accuracy for pretraining"""

        mask_pos = fluid.layers.cast(x=mask_pos, dtype='int32')

        # extract the first token feature in each sentence
        next_sent_feat = self.get_pooled_output(next_sent_index)
        # extract masked tokens' feature
        mask_feat = fluid.layers.gather(
            input=self._reshaped_emb_out, index=mask_pos)

        mask_lm_out_bias_attr = fluid.ParamAttr(
            name="mask_lm_out_bias",
            initializer=fluid.initializer.Constant(value=0.0))
        if self._weight_sharing:
            fc_out = fluid.layers.matmul(
                x=mask_feat,
                y=fluid.default_main_program().global_block().var(
                    self._word_emb_name),
                transpose_y=True)
            fc_out += fluid.layers.create_parameter(
                shape=[self._voc_size + 1],
                dtype="float32",
                attr=mask_lm_out_bias_attr,
                is_bias=True)

        else:
            fc_out = fluid.layers.fc(input=mask_feat,
                                     size=self._voc_size + 1,
                                     param_attr="mask_lm_out_w",
                                     bias_attr=mask_lm_out_bias_attr)

        mask_lm_loss = fluid.layers.softmax_with_cross_entropy(
            logits=fc_out, label=mask_label)
        mean_mask_lm_loss = fluid.layers.reduce_mean(mask_lm_loss)

        next_sent_fc_out = fluid.layers.fc(input=next_sent_feat,
                                           size=2,
                                           param_attr="next_sent_fc_w",
                                           bias_attr="next_sent_fc_b")

        next_sent_loss, next_sent_softmax = fluid.layers.softmax_with_cross_entropy(
            logits=next_sent_fc_out, label=labels, return_softmax=True)

        next_sent_acc = fluid.layers.accuracy(
            input=next_sent_softmax, label=labels)

        mean_next_sent_loss = fluid.layers.reduce_mean(next_sent_loss)

        loss = mean_next_sent_loss + mean_mask_lm_loss
        return next_sent_acc, mean_mask_lm_loss, loss
