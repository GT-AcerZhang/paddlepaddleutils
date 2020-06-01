import pandas as pd
# 划分验证集，保存格式  text[\t]label
from sklearn.model_selection import train_test_split


# 自定义数据集
import os
import codecs
import csv

from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset

def prepare_data():
    # 转换编码
    def re_encode(path):
        with open(path, 'r', encoding='GB2312', errors='ignore') as file:
            lines = file.readlines()
        with open(path, 'w', encoding='utf-8') as file:
            file.write(''.join(lines))
            
    #re_encode('data/test.csv')
    #re_encode('data/train.csv')

    # 读取数据
    train_labled = pd.read_csv('data/train.csv', engine ='python')
    test = pd.read_csv('data/test.csv', engine ='python')

    print(train_labled.shape)
    print(test.shape)
    print(train_labled.columns)

    train_labled.head(3)
    test.head(3)

    train_labled = train_labled[train_labled['情感倾向'].isin(['-1','0','1'])]
    train_labled['微博中文内容'].str.len().describe()


    train_labled = train_labled[['微博中文内容', '情感倾向']]
    train, valid = train_test_split(train_labled, test_size=0.2, random_state=2020)
    train.to_csv('./data/train.txt', index=False, header=False, sep='\t')
    valid.to_csv('./data/valid.txt', index=False, header=False, sep='\t')

class MyDataset(BaseNLPDataset):
    """DemoDataset"""
    def __init__(self):
        # 数据集存放位置
        self.dataset_dir = "./data"
        super(MyDataset, self).__init__(
            base_path=self.dataset_dir,
            train_file="train.txt",
            dev_file="valid.txt",
            test_file="valid.txt",
            train_file_with_header=False,
            dev_file_with_header=False,
            test_file_with_header=False,
            # 数据集类别集合
            label_list=["-1", "0", "1"])


def fine_tune():
    print("print my dataset")
    dataset = MyDataset()
    for e in dataset.get_train_examples()[:3]:
        print("{}\t{}\t{}".format(e.guid, e.text_a, e.label))

    import paddlehub as hub
    module = hub.Module(name="ernie", module_dir="./model")

    # 构建Reader
    print("vocab_path:", module.get_vocab_path())
    print("sp_model_path:", module.get_spm_path())
    print("word_dict_path:", module.get_word_dict_path())
    reader = hub.reader.ClassifyReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        sp_model_path=module.get_spm_path(),
        word_dict_path=module.get_word_dict_path(),
        max_seq_len=128)

    # finetune策略1
    strategy = hub.L2SPFinetuneStrategy(
        learning_rate=1e-4,
        optimizer_name="adam",
        regularization_coeff=1e-3)

    # 运行配置
    config = hub.RunConfig(
        use_cuda=True,
        num_epoch=10,
        batch_size=32,
        checkpoint_dir="hub_finetune_ckpt0431",
        strategy=strategy)

    # Finetune Task
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=128)

    # Use "pooled_output" for classification tasks on an entire sentence.
    pooled_output = outputs["pooled_output"]

    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    cls_task = hub.TextClassifierTask(
            data_reader=reader,
            feature=pooled_output,
            feed_list=feed_list,
            num_classes=dataset.num_labels,
            config=config,
            metrics_choices=["f1"])

    # finetune
    run_states = cls_task.finetune_and_eval()

if __name__ == "__main__":
    prepare_data()
    fine_tune()

    
