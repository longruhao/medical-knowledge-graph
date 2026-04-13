# -*- coding: utf-8 -*-
from config import *
from itertools import chain

conf = Config()

# 获取关系类型字典
rel2id = dict()
with open(conf.rel_data_path, 'r', encoding='utf-8') as fr:
    for line in fr.readlines():
        word, idx = line.strip().split()
        if word not in rel2id:
            rel2id[word] = int(idx)


def sent_padding(sent, word2id):
    '''
    把一个句子 word2id，并截断补齐
    :param sent: 一个句子 token 列表
    :param word2id: 词表：每个字符对应一个数字
    '''
    # 1 word2id
    ids = [word2id.get(word, word2id['UNK']) for word in sent]
    # 2 句子截断
    if len(ids) >= conf.max_length:
        return ids[:conf.max_length]
    # 3 句子补齐
    pad_len = conf.max_length - len(ids)
    ids.extend([word2id['PAD']] * pad_len)
    return ids


def pos(num):
    '''
    转换位置信息，因为 pos_embedding 不能出现负数，直接计算相对位置会有负数
    '''
    if num < -70:
        return 0
    elif (num >= -70) and (num <= 70):
        return num + 70
    else:  # > 70
        # 也可以是 140，只能 >= 140，142 考虑了句子的 起始符、结束符
        return 142


def pos_padding(pos_ids):
    '''
    使用 pos() 函数 把 pos位置信息 转为 非负 形式，并补全为 max_len 长度
    '''
    # 1 借助 pos 处理单个 pos_id
    pos_seq = [pos(pos_id) for pos_id in pos_ids]
    # 2 截断
    if len(pos_seq) >= conf.max_length:
        return pos_seq[:conf.max_length]
    # 3 补齐
    pad_len = conf.max_length - len(pos_seq)
    pos_seq.extend([142] * pad_len)
    return pos_seq


def get_data(data_path):
    '''
    将原始数据集格式转换，返回模型需要的数据格式
    '''
    # 1 定义存储容器
    # 存储每个样本句子 datas, 存储每个样本的实体对 ents, 存储每个样本的关系类型标签 labels
    # 存储每个样本中：每个字符距离第一个实体的相对位置信息 pos_e1, 每个字符距离第二个实体的相对位置信息 pos_e2
    datas = []
    ents = []
    pos_e1 = []
    pos_e2 = []
    labels = []

    # 每种关系的计数字典 初始值为0
    # count_dict = {key: 0 for key, value in rel2id.items()}

    # 2 读取数据 填充 datas, labels, pos_e1, pos_e2, ents
    # 2.1 循环遍历 data_path，将 line 切分为 line_list
    # line: '似水流年 许晓杰 作曲 似水流年，由著名作词家闫肃作词，著名音乐人许晓杰作曲，张烨演唱'
    for line in open(data_path, 'r', encoding='utf-8'):
        # maxsplit=3 切分从前往后的三个空格，原始文本中可能也有空格，不要切
        line_list = line.strip().split(' ', maxsplit=3)

        # 断言 len(line_list) 是否为 4
        assert len(line_list) == 4

        # 如果关系类型不在关系字典，直接跳过
        if line_list[2] not in rel2id:
            continue

        # 如果该关系类型样本超过2000条，直接跳过，保证了类别均衡
        # if count_dict[line_list[2]] > 2000:
        #     continue
        # else:  # 正常情况

        # 2.2 填充 ents
        ent1 = line_list[0]
        ent2 = line_list[1]
        ents.append([ent1, ent2])

        # 2.3 从 原始文本 找到两个实体的起始索引
        idx1 = line_list[3].index(ent1)
        idx2 = line_list[3].index(ent2)

        # 2.4 循环遍历原始句子 line_list[3]，获取句子序列 sent
        # 以及每个token相对实体idx的索引，并将索引保存到 pos1, pos2
        # 先初始化 sent, pos1, pos2
        sent = []
        pos1 = []
        pos2 = []

        for idx, word in enumerate(line_list[3]):
            sent.append(word)
            pos1.append(idx - idx1)
            pos2.append(idx - idx2)

        # 2.5 填充 datas、pos_e1、pos_e2、labels
        datas.append(sent)
        pos_e1.append(pos1)
        pos_e2.append(pos2)
        labels.append(rel2id[line_list[2]])  # 将关系类型进行数字转换
        # count_dict[line_list[2]] += 1

    return datas, labels, pos_e1, pos_e2, ents


def get_vocab(data_path):
    '''
    文本转id，得到word2id, id2word
    '''
    # 1 通过 get_data 拿到文本列表 datas [[], [], [],,,]
    datas = get_data(data_path)[0]

    # 2 扁平化 chain(*) 去重
    data_list = list(set(chain(*datas)))
    # print(f'data_list--->{data_list}')
    # print(f'data_list--->{len(data_list)}')

    # 3 构建 word2id, id2word
    word2id = {word: idx for idx, word in enumerate(data_list)}
    word2id['PAD'] = len(word2id)
    word2id['UNK'] = len(word2id)
    id2word = {v: k for k, v in word2id.items()}

    return word2id, id2word


# def get_vocab(data_path):
#     '''
#     文本转id，得到word2id, id2word
#     '''
#     # 1 通过 get_data 拿到文本列表 datas [[], [], [],,,]
#     datas = get_data(data_path)[0]
#
#     # 2 扁平化 chain(*) 去重
#     data_list = list(set(chain(*datas)))
#     data_list.extend(['PAD', 'UNK'])
#
#     # 3 保存词表
#     with open('./data/vocab.txt', 'w', encoding='utf-8') as fw:
#         fw.write('\n'.join(data_list))
#         fw.flush()


if __name__ == '__main__':
    data_path = conf.train_data_path

    # 测试 get_data()
    res = get_data(data_path)
    # print(res[0][0])
    # print(res[1][0])
    # print(res[2][0])
    # print(res[3][0])
    # print(res[4][0])
    # exit()

    word2id, id2word = get_vocab(data_path)
    print(f'word2id-->{word2id}')
    print(f'id2word-->{id2word}')
