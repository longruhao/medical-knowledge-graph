import json
import torch


class Config():
    '''
    定义常量
    '''

    def __init__(self):
        # 1 定义设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        # 2 数据路径
        # 训练集路径 train_path
        self.train_path = r'D:\workspace\code\projects\ai\knowledge_graph\ner_bi_lstm_crf\data\train.txt'
        # 词表路径 vocab_path
        self.vocab_path = r'D:\workspace\code\projects\ai\knowledge_graph\ner_bi_lstm_crf\data\vocab.txt'
        # 加载标签字典 tag2id
        self.tag2id = json.load(open(r'D:\workspace\code\projects\ai\knowledge_graph\ner_bi_lstm_crf\data\tag2id.json'))
        self.target = list(self.tag2id.keys())

        # 3 模型相关
        self.embedding_dim = 300
        self.hidden_dim = 256
        self.dropout = 0.2

        # 4 训练相关
        self.model = 'BiLSTM_CRF'
        # self.model = 'BiLSTM'
        self.epochs = 20
        self.batch_size = 16
        self.lr = 2e-4
        # self.crf_lr = 1e-3


if __name__ == '__main__':
    conf = Config()
    print(conf.train_path)
    print(conf.target)
