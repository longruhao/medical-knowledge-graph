import torch
import torch.nn as nn
from transformers import BertModel
from config import *
from utils.data_loader import *


class Casrel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        # 加载bert
        self.bert = BertModel.from_pretrained(conf.bert_path)
        # 识别主实体 开始索引，linear 可以接受三维输入
        self.sub_head_linear = nn.Linear(conf.bert_dim, 1)
        # 识别主实体 结束索引
        self.sub_tail_linear = nn.Linear(conf.bert_dim, 1)
        # 识别客实体 开始索引 和 关系类型
        self.obj_head_linear = nn.Linear(conf.bert_dim, conf.rel_class)
        # 识别客实体 结束索引 和 关系类型
        self.obj_tail_linear = nn.Linear(conf.bert_dim, conf.rel_class)

    def get_encode_reuslt(self, input_ids, attention_mask):
        # bert编码
        encoded_result = self.bert(input_ids, attention_mask)[0]
        return encoded_result

    def get_subs(self, encoded_result):
        '''
        用来预测主实体 开始索引 和 结束索引
        :param encoded_result:bert 编码结果
        '''
        pred_sub_heads = torch.sigmoid(self.sub_head_linear(encoded_result))
        pred_sub_tails = torch.sigmoid(self.sub_tail_linear(encoded_result))
        return pred_sub_heads, pred_sub_tails

    def get_objs_for_specific_sub(self, sub_head2tail, sub_len, encoded_text):
        '''
        融合 bert向量 + sub向量，主体信息加到每个token上，来预测 客体o、关系p
        sub_head2tail: shape = [2, 1, 200]，在一个样本句子中，随机挑选一个主体，作为预测 o p 的输入，1 表示 1 个实体
        sub_len: shape = [2, 1]，主体长度
        encoded_text: shape = [2, 200, 768]，bert编码的文本向量
        '''
        sub = torch.matmul(sub_head2tail, encoded_text)  # 拿到主实体向量 [2, 1, 768]
        sub_len = sub_len.unsqueeze(1)  # [2, 1, 1]
        sub = sub / sub_len  # 平均主实体信息
        encoded_text = encoded_text + sub  # 融合 bert向量 + sub向量
        # 预测 客体 开始索引、结束索引 以及 关系类型
        pred_obj_heads = torch.sigmoid(self.obj_head_linear(encoded_text))
        pred_obj_tails = torch.sigmoid(self.obj_tail_linear(encoded_text))
        return pred_obj_heads, pred_obj_tails

    def forward(self, input_ids, mask, sub_head2tail, sub_len):
        '''
        input_ids.shape = attention_mask.shape = (2,200)
        sub_head2tail.shape = (16,200), sub_len.shape = (16,1)
        '''
        encoded_result = self.get_encode_reuslt(input_ids, mask)
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_result)
        sub_head2tail = sub_head2tail.unsqueeze(1)  # (16, 1, 200)
        pred_obj_heads, pred_obj_tails = self.get_objs_for_specific_sub(sub_head2tail, sub_len, encoded_result)
        result_dict = {
            'pred_sub_heads': pred_sub_heads,
            'pred_sub_tails': pred_sub_tails,
            'pred_obj_heads': pred_obj_heads,
            'pred_obj_tails': pred_obj_tails,
            'mask': mask
        }
        return result_dict

    def compute_loss(self,
                     pred_sub_heads, pred_sub_tails,
                     pred_obj_heads, pred_obj_tails,
                     mask,
                     sub_heads, sub_tails,
                     obj_heads, obj_tails):
        '''
        计算损失，二分类交叉熵损失 必须满足 pred.shape = true.shape
        多分类交叉熵损失 满足 pre.shape (2,200,18) 对应 true.shape (2,200)，batch 匹配即可
        :param pred_sub_heads: [2, 200, 1]
        :param pred_sub_tails: [2, 200, 1]
        :param pred_obj_heads: [2, 200, 18]
        :param pred_obj_tails: [2, 200, 18]
        :param mask: shape-->[2, 200]
        :param sub_heads: shape-->[2, 200]
        :param sub_tails: shape-->[2, 200]
        :param obj_heads: shape-->[2, 200, 18]
        :param obj_tails: shape-->[2, 200, 18]
        '''
        rel_count = obj_heads.shape[-1]
        # rel_mask.shape = [2, 200, 18]
        rel_mask = mask.unsqueeze(-1).repeat(1, 1, rel_count)
        # 计算损失
        loss1 = self.loss(pred_sub_heads, sub_heads, mask)
        loss2 = self.loss(pred_sub_tails, sub_tails, mask)
        loss3 = self.loss(pred_obj_heads, obj_heads, rel_mask)
        loss4 = self.loss(pred_obj_tails, obj_tails, rel_mask)
        return loss1 + loss2 + loss3 + loss4

    def loss(self, pred, gold, mask):
        pred = pred.squeeze(-1)
        # reduction='none' 则 loss.shape 和 pred、gold 相同，反之求和
        loss = nn.BCELoss(reduction='none')(pred, gold)
        if loss.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        # loss * mask 哈德马积 只保留非 mask 部分的 loss
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss


def load_model(conf):
    device = conf.device
    casrel = Casrel(conf=conf).to(device)
    # 借助 BERT 做 fine_tuning，可以 L2 正则 权重衰减，防止过拟合
    # Loss' = Loss + λ⋅∣∣w∣∣^2，λ 是一个超参数，控制权重衰减的强度，||w||^2 是权重的L2范数
    # no_decay 中存放不进行权重衰减的参数name
    # 一般不对 偏置项、BN层 权重衰减：1 偏置项 用于调整输出，不直接影响学习复杂性
    # 2 规范化层 本身就是调整网络层输出分布，标准化 均值0 方差1，加速训练 提高鲁棒性，不需要权重衰减
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # 拿到 模型中的 参数 和 参数名字
    param_list = list(casrel.named_parameters())
    # any() 有一个为True，则返回True
    # 如果不在no_decay中，则进行权重衰减; 如果在no_decay中，则不进行权重衰减
    # weight_decay 就是 λ
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_list if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_list if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=conf.learning_rate)
    # 先不使用 lr 调度器 (lr 预热、退火)
    sheduler = None
    return casrel, optimizer, sheduler, device


if __name__ == '__main__':
    conf = Config()

    # todo 测试 get_encode_reuslt
    model = Casrel(conf)
    train_loader, dev_loader, test_loader = get_data()
    # train_loader 加载器、iter() 创建迭代器 可逐个访问batch
    inputs, labels = next(iter(train_loader))
    print(model.get_encode_reuslt(inputs['input_ids'], inputs['mask']).shape)

    # ---------------------------------------------
    # todo 测试 损失计算
    result = model(**inputs)
    result.update(labels)
    print(result.keys())
    loss = model.compute_loss(**result)
    print(loss.item())

    # ---------------------------------------------
    # load_model(conf)
