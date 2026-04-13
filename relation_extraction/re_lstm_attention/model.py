# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from data_loader import *


class BiLSTM_Attn(nn.Module):
    def __init__(self, conf, vocab_size, pos_size, tag_size):
        super().__init__()
        # 1 接收参数
        self.vocab_size = vocab_size
        self.embed_dim = conf.embed_dim
        self.pos_dim = conf.pos_dim
        self.hidden_dim = conf.hidden_dim
        self.pos_size = pos_size
        self.tag_size = tag_size
        self.batch_size = conf.batch_size

        # 2 定义网络层
        # 文本嵌入层
        self.word_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        # 位置编码层1 相对实体1的位置 embedding层
        self.pos1_embed = nn.Embedding(self.pos_size, self.pos_dim)
        # 位置编码层2
        self.pos2_embed = nn.Embedding(self.pos_size, self.pos_dim)
        # 双向LSTM层
        self.lstm = nn.LSTM(input_size=self.embed_dim + self.pos_dim * 2,
                            hidden_size=self.hidden_dim // 2,
                            bidirectional=True,
                            batch_first=True)
        # linear层
        self.linear = nn.Linear(self.hidden_dim, self.tag_size)
        # droput层
        self.dropout_embed = nn.Dropout(p=0.2)
        self.dropout_lstm = nn.Dropout(p=0.2)
        self.dropout_attn = nn.Dropout(p=0.2)
        # 注意力权重，nn.Parameter() 默认计算梯度
        self.attn = nn.Parameter(torch.randn(self.batch_size, 1, self.hidden_dim))

    def attention(self, H):
        # M.shape = H.shape = (batch_size, hidden_dim, seq_len)
        M = F.tanh(H)
        # self.atten.shape = (b_s, 1, h_d); a.shape = (b_s, 1, seq_len)
        a = F.softmax(torch.bmm(self.attn, M), dim=-1)
        # a.shape = (b_s, seq_len, 1)
        a = torch.transpose(a, 1, 2)
        # return shape = (b_s, h_d, 1)
        return torch.bmm(H, a)

    def forward(self, sent, pos1, pos2):
        word_embedding = self.word_embed(sent)
        pos1_embedding = self.pos1_embed(pos1)
        pos2_embedding = self.pos2_embed(pos2)
        input = torch.cat((word_embedding, pos1_embedding, pos2_embedding), 2)
        input = self.dropout_embed(input)
        lstm_out, _ = self.lstm(input)
        lstm_out = self.dropout_lstm(lstm_out)

        # (batch_size, seq_len, hidden_dim) --> (batch_size, hidden_dim, seq_len)
        lstm_out = lstm_out.permute(0, 2, 1)

        # self.attention 之后的 shape = (batch_size, hidden_dim, 1)
        attn_out = F.tanh(self.attention(lstm_out))

        # squeeze() 之后的shape = (b_s, h_d)
        attn_out = self.dropout_attn(attn_out).squeeze()

        # output.shape = (b_s, tag_size)
        output = self.linear(attn_out)

        return F.softmax(output)


if __name__ == '__main__':
    conf = Config()
    word2id, id2word = get_vocab(conf.train_data_path)
    vocab_size = len(word2id)
    pos_size = 143
    tag_size = len(rel2id)
    model = BiLSTM_Attn(conf, vocab_size, pos_size, tag_size)
    print(model)

    train_dataloader, test_dataloader = get_loader()
    sents_tensor, pos_e1_tensor, pos_e2_tensor, _, _, _, _ = next(iter(train_dataloader))
    out = model(sents_tensor, pos_e1_tensor, pos_e2_tensor)
    print(out.shape)
