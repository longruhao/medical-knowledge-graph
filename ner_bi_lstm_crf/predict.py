from model.BiLSTM import *
from model.BiLSTM_CRF import *
from utils.data_loader import *
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 准备 模型、id2tag
models = {'BiLSTM': NERLSTM, 'BiLSTM_CRF': NERLSTM_CRF}
model = models["BiLSTM_CRF"](conf.embedding_dim, conf.hidden_dim, conf.dropout, word2id, conf.tag2id)
model.load_state_dict(
    torch.load(r'D:\workspace\code\learning\knowledge_graph\day02\LSTM_CRF\save_model\bilstm_crf_best.pth',
               map_location=conf.device))

id2tag = {v: k for k, v in conf.tag2id.items()}


def model2test(sample):
    # 1 准备 x、mask
    x = [word2id.get(char, word2id["UNK"]) for char in sample]
    x_test = torch.tensor([x])  # 两维
    mask = (x_test != 0).long()

    # 2 开启测试模式，分模型预测
    model.eval()
    with torch.no_grad():
        if model.name == "BiLSTM":
            out = model(x_test, mask)
            pred = torch.argmax(out, dim=-1)
            tags = [id2tag[i.item()] for i in pred[0]]
        else:
            pred = model(x_test, mask)
            tags = [id2tag[i] for i in pred[0]]
        chars = list(sample)
        assert len(chars) == len(tags)
        res = extract_ents(chars, tags)
        return res


def extract_ents(chars, tags):
    # 1 初始化返回容器
    ents = []
    ent = []
    ent_type = None

    # 2 循环遍历 组装
    for char, tag in zip(chars, tags):
        # 2.1 如果 label 是 实体开始 B
        if tag.startswith("B"):
            # 如果已经有实体，先保存再清空
            if ent:
                # append() 1个元组 ()
                ents.append((ent_type, ''.join(ent)))
                ent = []
            ent_type = tag.split('-')[1]
            ent.append(char)
        # 2.2 如果 label 是 实体的中间和末尾 I
        elif tag.startswith("I") and ent:
            ent.append(char)
        # 2.3 label 为 O，执行 else
        else:
            # 如果已经有实体，先保存再清空
            if ent:
                ents.append((ent_type, ''.join(ent)))
                ent = []
                ent_type = None

    # # 如果最后一个实体没有保存，手动保存
    # if entity:
    #     entities.append((entity_type, ''.join(entity)))

    return {ent: ent_type for ent_type, ent in ents}


if __name__ == '__main__':
    text = '李华的父亲患有冠心病及糖尿病，无手术外伤史。'
    # text = '患者缘于3天前进食油腻食物后出现腹痛伴腹胀，以上腹部为著。偶有反酸，烧心，无恶心、呕吐，无头痛、头晕。'
    result = model2test(sample=text)
    print(result)
