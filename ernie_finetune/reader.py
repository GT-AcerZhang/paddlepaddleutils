import codecs
import os
import csv

from paddlehub.dataset import InputExample
from paddlehub.common.dir import DATA_HOME
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset

def space_tokenizer(i):
    return i.split()

class ChnSentiCorp(BaseNLPDataset):
    def __init__(self):
        base_path = "./data/"
        super(ChnSentiCorp, self).__init__(
            base_path=base_path,
            train_file="train.part.0",
            dev_file="dev.part.0",
            test_file="test.part.0",
            label_file=None,
            label_list=["0", "1"],
        )

        self._word_dict = None

    def __read_file(self, input_file):
        with codecs.open(input_file, "r", encoding="UTF-8") as f:
            for line in f:
                line = line.strip()
                if len(line) <= 0:
                    continue
                arr = line.split("\t")
                #print("line:", len(arr))
                yield arr


    def _read_file(self, input_file, phase=None):
        seq_id = 0
        examples = []
        for t in self.__read_file(input_file):
            if len(t) == 2:
                example = InputExample(
                    guid=seq_id, label=t[1], text_a=t[0])
                #print("t2", t[1])
            elif len(t) == 3:
                example = InputExample(
                    guid=seq_id, label=t[2], text_a=t[0])
                #print("t3", t[2])
            else:
                assert False, 'invalid format'
            seq_id += 1
            examples.append(example)

        return examples

    def student_word_dict(self, vocab_file):
        with codecs.open(vocab_file, "r", encoding="UTF-8") as f:
            self._word_dict = {i.strip(): l for l, i in enumerate(f.readlines())}

        return self._word_dict

    def student_reader(self, input_file, word_dict):
        """
        return [([seg_sentence_idxs], label, sentence), ()...]
        """
        def reader():
            for t in self.__read_file(input_file):
                s = []
                for word in space_tokenizer(t[1]):
                    idx = word_dict[word]  if word in word_dict else word_dict['[UNK]']
                    s.append(idx)
                yield s, t[2], t[0]

        return reader


if __name__ == '__main__':
    ds = ChnSentiCorp()
    ds._read_file("./data/train.part.0")
    ds.student_reader("./data/train.part.0", "./data/vocab.bow.txt")