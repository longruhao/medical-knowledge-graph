import torch
# Vocabulary 用于构建 rel2id 映射 `str` 到 `int`
# from fastNLP import Vocabulary
from transformers import BertTokenizer
import json


class Config(object):
    def __init__(self):
        # 1 设备 cuda mps cpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        # 2 bert相关：路径、tokenizer、输出维度
        self.bert_model_name = 'bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.bert_dim = 768

        # 3 数据集路径 绝对路径不会出错
        self.train_data = r'D:\workspace\code\projects\ai\knowledge_graph\relation_extraction\re_casrel\data\train.json'
        self.dev_data = r'D:\workspace\code\learning\knowledge_graph\day05\RE_CasRel\data\dev.json'
        self.test_data = r'D:\workspace\code\projects\ai\knowledge_graph\relation_extraction\re_casrel\data\test.json'
        self.rel_data = r'D:\workspace\code\projects\ai\knowledge_graph\relation_extraction\re_casrel\data\relation.json'

        # 4 关系类别相关：映射字典、类别数
        self.id2rel = json.load(open(self.rel_data, encoding='utf-8'))
        # self.rel_vocab = Vocabulary(padding=None, unknown=None)
        # self.rel_vocab.add_word_lst(list(self.id2rel.values()))
        self.rel2id = {v: int(k) for k, v in self.id2rel.items()}
        self.rel_class = 18  # 关系类别数

        # 5 训练相关超参数
        self.epochs = 10
        self.batch_size = 16
        self.learning_rate = 1e-5


if __name__ == '__main__':
    conf = Config()
    print(conf.id2rel)
    print(conf.rel_vocab)
    print(conf.rel_vocab.to_index('出生地'))
