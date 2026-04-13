# coding: utf-8
import os
import numpy as np
import torch
import torch.nn as nn


# todo 测试 bert、RoBERTa
def ceshi_bert():
    from transformers import AutoTokenizer, AutoModel
    # BERT
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('bert-base-chinese')
    # RoBERTa
    # tokenizer = AutoTokenizer.from_pretrained('./RoBERTa_zh_Large_PyTorch')
    # model = AutoModel.from_pretrained('./RoBERTa_zh_Large_PyTorch')
    text = ['太阳当空照', '花儿对我笑哈哈']
    data = tokenizer.batch_encode_plus(text, padding=True, return_tensors='pt')  # pt 返回的torch的tensor
    print(data.keys())
    print(data['input_ids'])  # (bs, seq_len)
    print(data['token_type_ids'])
    print(data['attention_mask'])
    out = model(data['input_ids'], data['attention_mask'])  # (2, seq_len, 768)
    print(out.keys())
    print(out['last_hidden_state'].shape)
    print(out['pooler_output'].shape)  # pooler_output 是 [cls]


# todo 测试 numpy 矩阵运算 np.matmul  @  np.dot
def ceshi_np_juzhen():
    '''
    np.matmul 等价于 @, 两者都不支持矩阵与标量乘法
    np.dot 支持矩阵标量乘法
    高维矩阵只操作后两位
    '''
    # 定义示例数组
    A = np.array([[1, 2], [3, 4]])  # 2x2 矩阵
    B = np.array([[5, 6], [7, 8]])  # 2x2 矩阵
    v = np.array([1, 2])  # 1D 向量

    # 使用 np.matmul()
    result_matmul_2d = np.matmul(A, B)  # 矩阵乘法
    result_matmul_1d = np.matmul(v, A)  # 1D 向量与矩阵的乘法

    # 使用 @ 运算符
    result_at_2d = A @ B  # 矩阵乘法
    result_at_1d = v @ A  # 1D 向量与矩阵的乘法

    # 使用 np.dot()
    result_dot_2d = np.dot(A, B)  # 矩阵乘法
    result_dot_1d = np.dot(v, A)  # 内积
    single = np.dot(A, 10)

    # 打印结果
    print("Result of np.matmul(A, B):\n", result_matmul_2d)
    print("Result of np.matmul(v, A):\n", result_matmul_1d)

    print("Result of A @ B:\n", result_at_2d)
    print("Result of v @ A:\n", result_at_1d)

    print("Result of np.dot(A, B):\n", result_dot_2d)
    print("Result of np.dot(v, A):\n", result_dot_1d)
    print("single:\n", single)


# todo 矩阵乘法 torch.mul、matmul、mm、bmm、dot、*、@
def ceshi_juzhen():
    # torch.mul() * 哈德马积(元素按位相乘)，支持广播
    # torch.matmul() @ 高维矩阵相乘(ab*bc=ac) 广播
    # torch.mm 二维矩阵乘法 不支持广播
    # torch.bmm 三维矩阵乘法 第一维 batch 不支持广播
    # mul * 哈德马积，支持广播，支持标量

    a = torch.randn(2, 3, 4)
    b = torch.randn(1, 3, 4)

    print(torch.mul(a, b).shape)
    print((a * b).shape)

    # matmul @ 高维矩阵运算，看最后两维，支持广播
    a = torch.randn(2, 3, 4)
    b = torch.randn(1, 4, 3)

    print(torch.matmul(a, b).shape)
    print((a @ b).shape)

    # mm 两维矩阵乘法，不支持广播
    a = torch.randn(3, 4)
    b = torch.randn(4, 3)
    print(torch.mm(a, b).shape)

    # bmm 三维矩阵乘法，第一维是 batch，不支持广播
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 4, 3)
    print(torch.bmm(a, b).shape)


# todo torch中的Tensor、Parameter、Variable有什么区别
def ceshi_tensor():
    a_array = np.random.randn(5, 3)
    a_tensor = torch.Tensor(a_array)
    print(a_tensor.requires_grad)

    a_vari = torch.autograd.Variable(a_tensor)
    print(a_vari.requires_grad)

    a_para = nn.Parameter(a_tensor)
    print(a_para.requires_grad)


# todo index
def ceshi_index():
    a_sent = '我爱北京天安门，天安门上太阳升。'
    a_str = '北京'
    print(a_sent.index(a_str))


# todo 测试 os.walk()
def ceshi_os_walk():
    origin_path = './LSTM_CRF/data_origin'
    num = 1
    for root, dirs, files in os.walk(origin_path):
        print(f'第{num}次遍历')
        num += 1
        print('root\t', root)
        print('dirs\t', dirs)
        print('files\t', files)
        print('*' * 50)


# todo 测试 dict.get()
def ceshi_get():
    a_dict = {4: 'B-LOC', 5: 'I-LOC', 6: 'I-LOC'}
    a_txt = '我爱北京天安门'
    res = []
    for idx, char in enumerate(a_txt):
        res.append(a_dict.get(idx, 'O'))
    print(res)


# todo 测试 repr
def ceshi_repr():
    print(repr('肿胀	29	30	症状和体征'))


def ceshi_biaozhu():
    a = ['以', '咳', '嗽', '，', '咳', '痰', '，', '发', '热', '为', '主', '症', '。']
    b = ['O', 'B-SIGNS', 'I-SIGNS', 'O', 'B-SIGNS', 'I-SIGNS',
         'O', 'B-SIGNS', 'I-SIGNS', 'O', 'O', 'O', 'O']
    for c, d in zip(a, b):
        print(c, d)


# todo 测试 pad_sequence
def ceshi_pad_sequence():
    import torch
    from torch.nn.utils.rnn import pad_sequence

    # 定义三个长度不等的句子
    seq1 = torch.tensor([1, 2, 3])
    seq2 = torch.tensor([4, 5])
    seq3 = torch.tensor([6, 7, 8, 9])
    # 放入一个列表: 列表嵌套tensor
    seqs = [seq1, seq2, seq3]

    # 使用 pad_sequence 进行填充
    padded_sequences = pad_sequence(seqs, batch_first=True, padding_value=0)
    print(padded_sequences)


# todo 测试 sys.path
def ceshi_sys_path():
    import sys

    print(sys.path)
    # module_path = "/path/to/your/module"
    # if module_path not in sys.path:
    #     sys.path.append(module_path)
    # 现在可以导入你自己的模块了
    # import your_module


# todo 测试 torch.argmax
def ceshi_torch_argmax():
    pred = torch.randn(2, 4, 5)
    pred = torch.argmax(pred, dim=-1)  # pres 三维变两维
    print(pred.shape)


# todo 测试 itertools.chain
def ceshi_chain():
    from itertools import chain
    list1 = [1, 2, 3]
    list2 = ['a', 'b', 'c']
    list3 = [True, False]
    combined = chain(list1, list2, list3)
    for element in combined:
        print(element)


# todo 测试 torch.stack()
def ceshi_stack():
    import torch
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])

    c = torch.randn(3, 4)
    d = torch.randn(3, 4)

    print(torch.stack([a, b]))
    print(torch.cat([a, b], dim=0).shape)
    print(torch.cat([c, d], dim=0).shape)
    print(torch.cat([c, d], dim=1).shape)


# todo 测试 defaultdict
def ceshi_defaultdict():
    import torch
    from collections import defaultdict  # 初始化一个字典
    s2ro_map = defaultdict(list)  # 指定为list，字典中所有的value都是list，key没有限制
    print(s2ro_map)
    s2ro_map['red'].append(2)
    s2ro_map['blue'] = [3]
    print(s2ro_map)
    # exit()

    a_defaultdict = defaultdict(list)
    a_defaultdict[(3, 5)] = [(7, 8, 0)]
    print(a_defaultdict)
    for one in a_defaultdict:
        print(one[0])
        print(one[1])
        print(one)
        print(a_defaultdict[one])


# todo 测试 fillna
def ceshi_fillna():
    import pandas as pd
    df = pd.DataFrame(columns=['TP', 'PRED', "REAL", 'p', 'r', 'f1'])
    print(df)
    df = pd.DataFrame(columns=['TP', 'PRED', "REAL", 'p', 'r', 'f1'], index=['sub', 'triple'])
    df.fillna(0, inplace=True)
    print(df)


# todo 测试 torch.stack()
def ceshi_stack():
    import torch
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    print(torch.stack([a, b]))


# todo 测试 拿到主实体向量
def ceshi_get_sub_vector():
    import torch
    encoded_text = torch.randn(2, 5, 5)  # 2个句子，5个token，5维向量
    sub_head2tail = torch.tensor(  # (2, 1, 5) 2个句子，1个实体，5个标记
        [
            [[0, 0, 1, 1, 1]],  # 标 1 的是实体位置
            [[1, 1, 0, 0, 0]],
        ],
        dtype=torch.float
    )
    print('sub_head2tail:\n', sub_head2tail)
    print('encoded_text:\n', encoded_text)
    print(torch.matmul(sub_head2tail, encoded_text))


# todo convert_score_to_zero_one
def ceshi_convert_score_to_zero_one(tensor):
    import torch
    tensor[tensor >= 0.5] = 1
    tensor[tensor < 0.5] = 0
    return tensor


# todo 测试 extract_sub
def ceshi_extract_sub():
    pred_sub_heads = torch.tensor([0, 0, 0, 1, 0])
    idx = torch.arange(0, len(pred_sub_heads))[pred_sub_heads == 1]
    print(torch.arange(0, len(pred_sub_heads)))
    print(pred_sub_heads == 1)
    print(idx.item())


if __name__ == '__main__':
    print('---------- Test Code ----------')
    # ceshi_get()
    # ceshi_os_walk()
    # ceshi_repr()
    # ceshi_biaozhu()
    # ceshi_pred()
    # ceshi_np_juzhen()
    # ceshi_stack()
    # ceshi_tensor()
    # ceshi_extract_sub()
    ceshi_bert()
    # ceshi_get_sub_vector()
