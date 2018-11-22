import logging

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.transpiler.details import program_to_code

from utils import to_lodtensor

def mask(batch_tokens, total_token_num, vocab_size, CLS=1, SEP=2, MASK=0):
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    """
    max_len = max([len(sent) for sent in batch_tokens])
    mask_label = []
    mask_pos = []
    prob_mask  = np.random.rand(total_token_num)
    # Note: the first token is [CLS], so [low=1]
    replace_ids = np.random.randint(1, high=vocab_size, size=total_token_num)
    pre_sent_len = 0
    prob_index = 0
    for sent_index, sent in enumerate(batch_tokens):
        mask_flag = False
        prob_index += pre_sent_len
        for token_index, token in enumerate(sent):
            prob = prob_mask[prob_index + token_index]
            if  prob > 0.15:
                continue
            elif 0.03 < prob <= 0.15:
                # mask
                if token != SEP and token != CLS:
                    mask_label.append(sent[token_index])
                    sent[token_index] = MASK
                    mask_flag = True
                    mask_pos.append(sent_index * max_len + token_index)
            elif 0.015 < prob <= 0.03:
                # random replace
                if token != SEP and token != CLS:
                    mask_label.append(sent[token_index])
                    sent[token_index] = replace_ids[prob_index + token_index]
                    mask_flag = True
                    mask_pos.append(sent_index * max_len + token_index)
            else:
                # keep the original token
                if token != SEP and token != CLS:
                    mask_label.append(sent[token_index])
                    mask_pos.append(sent_index * max_len + token_index)
        pre_sent_len = len(sent)

        # ensure at least mask one word in a sentence
        while not mask_flag:
            token_index = int(np.random.randint(1, high=len(sent)-1, size=1))
            if sent[token_index] != SEP and sent[token_index] != CLS:
                mask_label.append(sent[token_index])
                sent[token_index] = MASK
                mask_flag = True
                mask_pos.append(sent_index * max_len + token_index)
    mask_label = np.array(mask_label).astype("int64").reshape([-1,1])
    mask_pos = np.array(mask_pos).astype("int64").reshape([-1,1])
    return batch_tokens, mask_label, mask_pos

def prepare_batch_data(insts,
                   total_token_num,
                   n_head=1,
                   voc_size=0,
                   pad_word_id=0,
                   pad_pos_id=512,
                   pad_sent_id=3,
                   mask_id=0,
                   return_attn_bias=True,
                   return_max_len=True,
                   place=fluid.CPUPlace(),
                   return_num_token=False):
    """
    1. generate LodTensor of data
    2. generate LodTensor of position
    3. generate self attention mask, [shape: batch_size * n_head * max_len * max_len]
    """
    
    batch_src_ids  = [inst[0] for inst in insts]
    batch_sent_ids = [inst[1] for inst in insts]
    batch_pos_ids  = [inst[2] for inst in insts]
    labels  = [inst[3] for inst in insts]
    labels = np.array(labels).astype("int64").reshape([-1,1])

    # First step: do mask without padding
    out, mask_label, mask_pos = mask(batch_src_ids, total_token_num, vocab_size=voc_size, CLS=1, SEP=2, MASK=mask_id)
    
    # Second step: padding
    src_id, next_sent_index, self_attn_bias = pad_batch_data(out, n_head=n_head, pad_idx=pad_word_id, return_next_sent_pos=True, return_attn_bias=True)
    pos_id  = pad_batch_data(batch_pos_ids, n_head=n_head, pad_idx=pad_pos_id, return_pos=False, return_attn_bias=False)
    sent_id  = pad_batch_data(batch_sent_ids, n_head=n_head, pad_idx=pad_sent_id, return_pos=False, return_attn_bias=False)

    return_list = [src_id, pos_id, sent_id, self_attn_bias, mask_label, mask_pos, labels, next_sent_index]

    return return_list if len(return_list) > 1 else return_list[0]


def pad_batch_data(insts,
                   pad_idx=0,
                   n_head=1,
                   return_pos=False,
                   return_next_sent_pos=False,
                   return_attn_bias=False,
                   return_max_len=False,
                   return_num_token=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    # next_sent_pos for extract first token embedding of each sentence
    if return_next_sent_pos:
        batch_size = inst_data.shape[0]
        max_seq_len = inst_data.shape[1]
        next_sent_index = np.array(range(0, batch_size * max_seq_len, max_seq_len)).astype("int64").reshape(-1, 1)
        return_list += [next_sent_index]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
         for inst in insts
      ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_attn_bias:
        # This is used to avoid attention on paddings.
        slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                       (max_len - len(inst))
                                       for inst in insts])
        slf_attn_bias_data = np.tile(
            slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
            [1, n_head, max_len, 1])
        return_list += [slf_attn_bias_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    return return_list if len(return_list) > 1 else return_list[0]


if __name__ == "__main__":
    pass
