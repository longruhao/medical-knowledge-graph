from config import *
import torch
from random import choice
from collections import defaultdict

conf = Config()


def find_head_index(source, target):
    '''
    查找句子中实体的开始索引，如果没有返回 -1
    :param source: 原始句子 source = [101, 23, 45, 65, 76, 102]
    :param target: 实体 target = [45, 65]
    '''
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def create_label(inner_triples, inner_input_ids, seq_len):
    '''
    获取 主实体长度、主实体开始和结束位置张量表示、客实体以及对应关系实现张量表示
    :param inner_triples: 一个样本的 spo_list
    :param inner_input_ids: 一个文本 bert-tokenzier 之后结果
    :param seq_len: 文本长度 一个 batch 内的文本长度相等
    :return:
    '''
    # 1 label 初始化
    inner_sub_heads, inner_sub_tails = torch.zeros(seq_len), torch.zeros(seq_len)
    inner_obj_heads = torch.zeros((seq_len, conf.rel_class))
    inner_obj_tails = torch.zeros((seq_len, conf.rel_class))
    # inner_sub_head2tail 随机抽取一个主体，拿到 开始、结束 索引 (s, p, o)
    inner_sub_head2tail = torch.zeros(seq_len)

    # inner_sub_len 实体长度，暂时初始化为1，防止除0报错
    inner_sub_len = torch.tensor([1], dtype=torch.float)
    # s2ro_map s主体 到 (r关系、o客体) 的映射，这种写法 value 必须是 list
    s2ro_map = defaultdict(list)
    # print(s2ro_map)  # {'red':[12, 23, 34],,,,,}
    
    # 2  2.1 循环遍历 解析 inner_triples
    for inner_triple in inner_triples:
        # 2.2 获取 头实体、关系类别、尾实体 的 id 映射
        inner_triple = (
            conf.tokenizer(inner_triple['subject'], add_special_tokens=False)['input_ids'],
            # (input_ids, token_type_ids, attention_mask)
            conf.rel2id[inner_triple['predicate']],  # 获取关系id
            conf.tokenizer(inner_triple['object'], add_special_tokens=False)['input_ids']
        )
        # 2.3 获取 主实体、客实体 的 起始位置索引
        # eg：inner_input_ids = [101, 23, 45, 65, 76, 102]
        # inner_triple[0] = [45, 65]
        # find_head_index 寻找 inner_triple[0] 在 inner_input_ids 中的 起始位置，即 2
        sub_head_idx = find_head_index(source=inner_input_ids, target=inner_triple[0])
        obj_head_idx = find_head_index(source=inner_input_ids, target=inner_triple[2])

        # 2.4 填充 s2ro_map 字典
        # !-1 表示 主体、客体 都存在
        if sub_head_idx != -1 and obj_head_idx != -1:
            # 获取 头实体 开始位置、结束位置 的 索引
            sub = (sub_head_idx, sub_head_idx + len(inner_triple[0]) - 1)
            # s2ro_map 保存 主实体 到 (尾实体、关系) 的映射，eg：# {(3, 5): [(7, 8, 0)]}
            s2ro_map[sub].append(
                (obj_head_idx, obj_head_idx + len(inner_triple[2]) - 1, inner_triple[1])
            )

    # 3 填充工作
    if s2ro_map:
        # s2ro_map = { (3, 5): [(7, 8, 0)],  (2, 17): [(22, 28, 11)] }
        # 则 s = (3, 5), s[0] = 3, s[1] = 5, s2ro_map[s] = [(7, 8, 0)]
        # 3.1 填充 预测主体 label
        for s in s2ro_map:
            # 主体 开始、结束 位置赋值为1，类似独热编码
            inner_sub_heads[s[0]] = 1  # [0, 0, 0, 1, 0, 0, 0, 0]
            inner_sub_tails[s[1]] = 1  # [0, 0, 0, 0, 0, 1, 0, 0]
        # 3.2 填充 预测 obj-rel 的输入 inner_sub_head2tail
        # 随机选择一个主体，拿到 开始、结束 索引，将两个索引间的值赋值为 1，并计算实体长度
        sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
        inner_sub_head2tail[sub_head_idx: sub_tail_idx + 1] = 1

        # 3.3 填充 inner_sub_len
        inner_sub_len = torch.tensor([sub_tail_idx + 1 - sub_head_idx], dtype=torch.float)

        # 3.4 根据实体 拿到 obj-rel label
        # s2ro_map.get((3,5)) = [(7,8,0), (12,15,6)]
        # ro[0] = 7 客体起始索引, ro[1] = 8 客体结束索引, ro[2] = 0 关系类型
        for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
            inner_obj_heads[ro[0]][ro[2]] = 1  # ro[0] ro[1] 确定谁是客体
            inner_obj_tails[ro[1]][ro[2]] = 1  # ro[2] 确定谁是关系

    return (inner_sub_len, inner_sub_head2tail, inner_sub_heads,
            inner_sub_tails, inner_obj_heads, inner_obj_tails)


def collate_fn(data):
    '''
    :param data: 假设 data 的长度:2，代表一个批次 2 个样本，每个样本是: (text, spo_list)
    "text": "人物生平郭静唐，1903年2月4日出生，又名挹青、一青、澄，字琴堂，周巷镇徐家荒场人"
    "spo_list": [
    {"predicate": "出生地", "object_type": "地点", "subject_type": "人物", "object": "周巷", "subject": "郭静唐"},
    {"predicate": "出生日期", "object_type": "日期", "subject_type": "人物", "object": "1903年2月4日", "subject": "郭静唐"}
    ]  # 列表中有几个json 就有几个spo三元组
    :return:
    '''
    # 1 拆分数据
    text_list = [value[0] for value in data]
    triple = [value[1] for value in data]
    # print(text_list)
    # print(triple)

    # 2 编码句子，获取 batch_size seq_len
    # 对 1 个 batch 的 text 进行 bert-tokenizer 编码，padding=True 按照 batch 中最长句子补齐
    text = conf.tokenizer.batch_encode_plus(text_list, padding=True)
    # print(text.keys())  # ['input_ids', 'token_type_ids', 'attention_mask']
    # attention_mask padding 部分置为0，其余位置为1

    batch_size = len(text['input_ids'])  # 防止 drop_last=False
    # 一个批次样本的编码长度相同，不同批次长度可能不一样
    seq_len = len(text['input_ids'][0])

    # 3 初始化返回容器
    sub_heads = []  # 保存主体的头索引
    sub_tails = []
    obj_heads = []
    obj_tails = []
    sub_len = []  # 保存主体的长度
    sub_head2tail = []  # 整个主体span全部赋值为1，并且这个sub是随机抽取的，作为训练 obj-rel 的输入

    # 4 循环遍历每条样本，使用 create_label 组装结果
    for batch_idx in range(batch_size):
        inner_triples = triple[batch_idx]  # 一条样本的全部spo三元组
        inner_input_ids = text['input_ids'][batch_idx]  # 一条样本的文本编码结果

        # print('inner_triples:', inner_triples)
        # print('inner_input_ids:', inner_input_ids)
        # exit()
        '''
        inner_triples: [
        {'predicate': '出生地', 'object_type': '地点', 'subject_type': '人物', 'object': '广东', 'subject': '陈列'}, 
        {'predicate': '出生日期', 'object_type': '日期', 'subject_type': '人物', 'object': '1967年3月', 'subject': '陈列'}, 
        {'predicate': '民族', 'object_type': '文本', 'subject_type': '人物', 'object': '汉族', 'subject': '陈列'}
        ]
        inner_input_ids: [
        101, 7357, 1154, 8024, 4511, 8024, 9128, 2399, 124, 3299, 4495, 8024, 3727, 3184, 8024, 
        5093, 6581, 510, 1139, 4495, 1765, 2408, 691, 1426, 2335, 8024, 8431, 2399, 129, 3299, 1346, 1217, 2339,
         868, 8024, 8431, 2399, 128, 3299, 1057, 1054, 8024, 704, 1925, 1054, 3413, 1762, 5466, 4777, 4955, 4495, 
         2110, 1325, 8020, 704, 1925, 1054, 3413, 5307, 3845, 2110, 683, 689, 8021, 102, 0, 0, 0, 0, 0, 0, 0, 0, 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
         ]
        '''

        results = create_label(inner_triples, inner_input_ids, seq_len)
        sub_len.append(results[0])
        sub_head2tail.append(results[1])
        sub_heads.append(results[2])
        sub_tails.append(results[3])
        obj_heads.append(results[4])
        obj_tails.append(results[5])

    input_ids = torch.tensor(text['input_ids']).to(conf.device)
    mask = torch.tensor(text['attention_mask']).to(conf.device)

    # 5 借助torch.stack()函数沿一个新维度对输入batch_size张量序列进行连接
    sub_heads = torch.stack(sub_heads).to(conf.device)
    sub_tails = torch.stack(sub_tails).to(conf.device)
    sub_len = torch.stack(sub_len).to(conf.device)
    sub_head2tail = torch.stack(sub_head2tail).to(conf.device)
    obj_heads = torch.stack(obj_heads).to(conf.device)
    obj_tails = torch.stack(obj_tails).to(conf.device)

    inputs = {
        'input_ids': input_ids,
        'mask': mask,
        'sub_head2tail': sub_head2tail,
        'sub_len': sub_len
    }
    labels = {
        'sub_heads': sub_heads,
        'sub_tails': sub_tails,
        'obj_heads': obj_heads,
        'obj_tails': obj_tails
    }

    return inputs, labels


def extract_sub(pred_sub_heads, pred_sub_tails):
    '''
    pred_sub_heads: 预测的 主实体 头索引 shape = [seq_len]
    pred_sub_tails: 预测的 主实体 尾索引 shape = [seq_len]
    :return: subs 主实体的头尾索引 [head, tail]
    '''
    subs = []
    # todo 查看测试代码
    heads = torch.arange(0, len(pred_sub_heads), device=conf.device)[pred_sub_heads == 1]
    tails = torch.arange(0, len(pred_sub_tails), device=conf.device)[pred_sub_tails == 1]
    for head, tail in zip(heads, tails):  # 即使 len(heads) != len(tails)，也不会出现异常，按位组合
        if tail >= head:
            subs.append((head.item(), tail.item()))
    return subs


def extract_obj_and_rel(obj_heads, obj_tails):
    '''
    :param obj_heads: 预测 客体 开头位置 以及 关系类型 shape=[70, 18]
    :param obj_tails: 预测 客体 尾部位置 以及 关系类型 shape=[70, 18]
    :return: obj_and_rels：shape = [(rel_index, start_index, end_index),,,]
    '''
    obj_heads = obj_heads.T  # (18, 70)
    obj_tails = obj_tails.T
    rel_count = obj_heads.shape[0]
    obj_and_rel = []
    for rel_index in range(rel_count):  # 循环每种关系，即关系索引
        obj_head = obj_heads[rel_index]
        obj_tail = obj_tails[rel_index]
        # 借助 extract_sub() 识别 客体 的头尾索引，shape 为 [[head, tail], [head, tail],,,]
        objs = extract_sub(obj_head, obj_tail)
        if objs:
            for obj in objs:
                start_index, end_index = obj
                obj_and_rel.append((rel_index, start_index, end_index))
    return obj_and_rel


def convert_score_to_zero_one(tensor):
    '''
    以0.5为阈值，大于0.5的设置为1，小于0.5的设置为0
    '''
    tensor[tensor >= 0.5] = 1
    tensor[tensor < 0.5] = 0
    return tensor
