import json
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.common import *

# 获取(x,y)数据对、word2id词表
datas, word2id = build_data()


class NerDataset(Dataset):
    def __init__(self, datas):
        super().__init__()
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        x = self.datas[item][0]  # [0]表示sample_x
        y = self.datas[item][1]  # [1]表示sample_y
        return x, y


def collate_fn(batch):
    # 1 拿到1个batch的x_train、y_train，并用word2id、conf.tag2id做映射为id
    # x_train、y_train 格式为 [tensor0, tensor1, tensor2,,,]
    # 为什么列表嵌套tensor，方便pad_sequence补齐
    x_train = [torch.tensor([word2id[char] for char in data[0]]) for data in batch]
    y_train = [torch.tensor([conf.tag2id[label] for label in data[1]]) for data in batch]
    # 2 使用pad_sequence补齐 todo
    # x_train填充0 --> input_ids_pad
    # y_train填充11 --> labels_pad
    input_ids_pad = pad_sequence(x_train, batch_first=True, padding_value=0)
    labels_pad = pad_sequence(y_train, batch_first=True, padding_value=11)
    # labels_pad = pad_sequence(y_train, batch_first=True, padding_value=0)
    # 3 创建attention mask
    attention_mask = (input_ids_pad != 0).long()
    return input_ids_pad, labels_pad, attention_mask


def get_data():
    # 总样本7836，datas[:6200]占比约80%
    train_dataset = NerDataset(datas[:6200])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=conf.batch_size,
                              collate_fn=collate_fn,
                              drop_last=True)

    dev_dataset = NerDataset(datas[6200:])
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=conf.batch_size,
                            collate_fn=collate_fn,
                            drop_last=True)
    return train_loader, dev_loader


if __name__ == '__main__':
    print(len(datas))
    train_dataloader, dev_dataloader = get_data()
    # train_dataloader数据加载器，是1个可迭代对象，但不是迭代器
    # iter()创建迭代器，支持逐个访问，next()拿到下一个元素
    inputs, labels, mask = next(iter(train_dataloader))
    print(inputs.shape)
    print(labels.shape)
    print(mask.shape)
    print(inputs[0])
    print(labels[0])
    print(mask[0])
