import torch
import torch.nn as nn
from config import *
from utils.data_loader import *

conf = Config()


class NERLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id):  # 1 初始化
        super().__init__()
        # 1 接收参数 name、embedding_dim、hidden_dim、dropout、word2id、tag2id
        self.name = "BiLSTM"
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id)
        self.tag2id = tag2id
        self.tag_size = len(tag2id)

        # 2 实例化网络层 word_embeds、lstm、dropout、hidden2tag(linear)
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)

    def forward(self, x, mask):  # 2 前向传播
        out = self.word_embeds(x)
        out, _ = self.lstm(out)
        out = out * mask.unsqueeze(-1)  # * 哈德马积
        out = self.dropout(out)
        out = self.hidden2tag(out)
        return out


if __name__ == '__main__':
    train_loader, dev_loader = get_data()
    inputs, labels, mask = next(iter(train_loader))
    print('inputs.shape:\t', inputs.shape)
    print('labels.shape:\t', labels.shape)

    embedding_dim = conf.embedding_dim
    hidden_dim = conf.hidden_dim
    dropout = conf.dropout
    tag2id = conf.tag2id
    model = NERLSTM(embedding_dim, hidden_dim, dropout, word2id, tag2id)
    print(model(inputs, mask).shape)
