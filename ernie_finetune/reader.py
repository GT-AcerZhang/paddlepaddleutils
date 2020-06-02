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
        pass
        

    def __read_file(self, input_file):
        with codecs.open(input_file, "r", encoding="UTF-8") as f:
            for line in f:
                line = line.strip()
                if len(line) <= 0:
                    continue
                arr = line.split("\t")
                assert len(arr) == 3
                yield arr


    def _read_file(self, input_file, phase=None):
        seq_id = 0
        examples = []
        for t in self.__read_file(input_file):
            example = InputExample(
                    guid=seq_id, label=t[2], text_a=t[0])
            seq_id += 1
            examples.append(example)

            return examples

    def student_reader(self, input_file, vocab_file):
        with codecs.open(vocab_file, "r", encoding="UTF-8") as f:
            student_vocab = {i.strip(): l for l, i in enumerate(f.readlines())}

        #print("student_vocab", student_vocab)
        r = []
        for t in self.__read_file(input_file):
            s = []
            for word in space_tokenizer(t[1]):
                idx = student_vocab[word]  if word in student_vocab else student_vocab['[UNK]']
                s.append(idx)
                #print("word_idx:", idx)
            r.append((s, t[2]))

        return r


if __name__ == '__main__':
    ds = ChnSentiCorp()
    ds._read_file("./data/train/part.0")
    ds.student_reader("./data/train/part.0", "./data/vocab.bow.txt")
