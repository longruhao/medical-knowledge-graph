from torch.utils.data import DataLoader, Dataset
from pprint import pprint  # 美化输出 层次结构输出
from config import *
from utils.process import *

conf = Config()


class MyDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.dataset = [json.loads(line) for line in open(data_path, encoding='utf-8')]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        content = self.dataset[item]
        text = content['text']
        spo_list = content['spo_list']
        return text, spo_list


def get_data():
    # 实例化dataset
    train_data = MyDataset(conf.train_data)
    dev_data = MyDataset(conf.dev_data)
    test_data = MyDataset(conf.test_data)

    # 实例化dataloader
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=conf.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    dev_dataloader = DataLoader(
        dataset=dev_data,
        batch_size=conf.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=conf.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return train_dataloader, dev_dataloader, test_dataloader


if __name__ == '__main__':
    train_data = MyDataset(conf.train_data)
    print(f'train_data[0]{train_data[0]}')

    train_loader, dev_loader, test_loader = get_data()
    inputs, labels = next(iter(train_loader))
    print(inputs)
    print(labels)
