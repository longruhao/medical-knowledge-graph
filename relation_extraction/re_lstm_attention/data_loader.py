# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader, Dataset
from process import *


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data = get_data(data_path)

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, i):
        sent = self.data[0][i]
        label = self.data[1][i]
        pos_e1 = self.data[2][i]
        pos_e2 = self.data[3][i]
        ent = self.data[4][i]
        return sent, label, pos_e1, pos_e2, ent


def collate_fn(datas):
    # datas是一个批次的样本：比如batch_size=4, datas代表4个样本数据
    # 每个样本数据包含：(sent, label, pos_e1, pos_e2, ent)
    # 1 拆解datas，分别拿到sents labels pos_e1 pos_e2 ents
    sents = [data[0] for data in datas]
    labels = [data[1] for data in datas]
    pos_e1 = [data[2] for data in datas]
    pos_e2 = [data[3] for data in datas]
    ents = [data[4] for data in datas]

    # 2 获取 word2id, id2word
    # word2id, id2word = get_vocab(conf.train_data_path)

    text_list = open('./data/vocab.txt', 'r', encoding='utf-8').read().strip().split('\n')
    word2id = {char: idx for idx, char in enumerate(text_list)}

    # 3 分别初始化并填充 sents_ids pos_e1_ids pos_e2_ids
    sents_ids = []
    for sent in sents:
        ids = sent_padding(sent, word2id)
        sents_ids.append(ids)

    pos_e1_ids = []
    for pos_ids in pos_e1:
        pos_ids1 = pos_padding(pos_ids)
        pos_e1_ids.append(pos_ids1)

    pos_e2_ids = []
    for pos_ids in pos_e2:
        pos_ids2 = pos_padding(pos_ids)
        pos_e2_ids.append(pos_ids2)

    # 4 转换 tensor
    sents_tensor = torch.tensor(sents_ids, dtype=torch.long)
    pos_e1_tensor = torch.tensor(pos_e1_ids, dtype=torch.long)
    pos_e2_tensor = torch.tensor(pos_e2_ids, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return sents_tensor, pos_e1_tensor, pos_e2_tensor, labels_tensor, sents, labels, ents


def get_loader():
    train_data = MyDataset(conf.train_data_path)
    train_loader = DataLoader(dataset=train_data,
                              shuffle=True,
                              batch_size=conf.batch_size,
                              drop_last=True,  # drop_last 对批归一化有好处，有效计算均值方差，测试阶段不需要
                              collate_fn=collate_fn)

    test_data = MyDataset(conf.test_data_path)
    test_loader = DataLoader(dataset=test_data,
                             shuffle=True,
                             batch_size=conf.batch_size,
                             drop_last=True,
                             collate_fn=collate_fn)

    return train_loader, test_loader


if __name__ == '__main__':
    # 测试 MyDataset
    dataset = MyDataset(conf.train_data_path)
    print('dataset[0]:', dataset[0])

    # 测试 get_data_loader
    train_dataloader, test_dataloader = get_loader()
    print('train_dataloader:', len(train_dataloader))
    print('test_dataloader:', len(test_dataloader))
    sents_tensor, pos_e1_tensor, pos_e2_tensor, labels_tensor, _, _, _ = next(iter(train_dataloader))
    print('sents_tensor:', sents_tensor.shape)
    print('pos_e1_tensor:', pos_e1_tensor.shape)
    print('labels_tensor:', labels_tensor.shape)
