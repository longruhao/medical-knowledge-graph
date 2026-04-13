import torch
import torch.nn as nn
# pip install pytorch-crf==0.7.2
from torchcrf import CRF
from utils.data_loader import *


class NERLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id):  # 1 初始化
        super().__init__()
        # 1 接收参数 name、embedding_dim、hidden_dim、vocab_size、tag_to_ix、tag_size
        self.name = "BiLSTM_CRF"
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id)
        self.tag2id = tag2id
        self.tag_size = len(tag2id)

        # 2 实例化网络层 word_embeds、lstm、dropout、hidden2tag、crf
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.crf = CRF(self.tag_size, batch_first=True)

    def get_lstm2linear(self, x):  # 2 拿到 发射分数：lstm-linear result
        out = self.word_embeds(x)
        out, _ = self.lstm(out)
        out = self.dropout(out)
        out = self.hidden2tag(out)
        return out

    def forward(self, x, mask):  # 3 解码最优路径 前向传播
        '''
        crf 需要 mask.bool()
        解码用 crf.decode() 解码结果是list，不是tensor
        计算损失用 crf 前向传播
        '''
        out = self.get_lstm2linear(x)
        out = out * mask.unsqueeze(-1)
        out = self.crf.decode(out, mask.bool())
        return out

    def log_likelihood(self, x, tags, mask):  # 4 计算损失
        out = self.get_lstm2linear(x)
        out = out * mask.unsqueeze(-1)
        return -self.crf(out, tags, mask.bool(), reduction='mean')


if __name__ == '__main__':
    train_dataloader, dev_dataloader = get_data()
    inputs, labels, mask = next(iter(train_dataloader))

    embedding_dim = conf.embedding_dim
    hidden_dim = conf.hidden_dim
    dropout = conf.dropout
    tag2id = conf.tag2id
    model = NERLSTM_CRF(embedding_dim, hidden_dim, dropout, word2id, tag2id)

    # 解码1条最优路径
    path = model(inputs, mask)[0]
    id2tag = {v: k for k, v in conf.tag2id.items()}
    path = [id2tag[i] for i in path]

    # 计算损失
    loss = model.log_likelihood(inputs, labels, mask).item()
    print('最优路径：%s\n平均损失：%.4f' % (path, loss))
