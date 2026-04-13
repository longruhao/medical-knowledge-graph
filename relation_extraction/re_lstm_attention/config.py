# -*- coding: utf-8 -*-
import torch


class Config():
    def __init__(self):
        # 1 定义设备
        # self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 2 数据集路径 windows vs linux
        # self.train_data_path = r'D:\ITCAST\课件\人工智能\AI项目\泛娱乐数据关系抽取\Bilstm_Attention_RE\data\train.txt'
        # self.test_data_path = r'D:\ITCAST\课件\人工智能\AI项目\泛娱乐数据关系抽取\Bilstm_Attention_RE\data\test.txt'
        # self.rel_data_path = r'D:\ITCAST\课件\人工智能\AI项目\泛娱乐数据关系抽取\Bilstm_Attention_RE\data\relation2id.txt'

        self.train_data_path = r'D:\workspace\code\projects\ai\knowledge_graph\relation_extraction\re_lstm_attention\data\train.txt'
        self.test_data_path = r'D:\workspace\code\projects\ai\knowledge_graph\relation_extraction\re_lstm_attention\data\test.txt'
        self.rel_data_path = r'D:\workspace\code\projects\ai\knowledge_graph\relation_extraction\re_lstm_attention\data\relation2id.txt'

        # 3 模型参数
        # 词嵌入维度
        self.embed_dim = 128
        # 位置嵌入维度 相对位置编码的嵌入维度
        self.pos_dim = 32
        # BiLSTM 隐藏层维度
        self.hidden_dim = 200
        # 句子最大长度 经过句子长度分布分析而来
        self.max_length = 70

        # 4 训练参数
        self.epochs = 60
        self.batch_size = 256
        self.learning_rate = 1e-3


if __name__ == '__main__':
    # conf = Config()
    # print(conf.train_data_path)
    # print(conf.batch_size)

    # 普通字符串中，反斜杠 \ 表示转义，\t \n \\
    # 想表示原始字符串：
    # 1 开头加 r，表示不去转义，仍为原始字符串
    # 2 Python 中双反斜杠 表示 单反斜杠，\\
    print('D:\ITCAST\课件\人工智能\AI项目\泛娱乐数据关系抽取\Bilstm_Attention_RE\data\train.txt')
    print('D:\\ITCAST\\课件\\人工智能\\AI项目\\泛娱乐数据关系抽取\\Bilstm_Attention_RE\\data\\train.txt')
    print(r'D:\ITCAST\课件\人工智能\AI项目\泛娱乐数据关系抽取\Bilstm_Attention_RE\data\train.txt')
