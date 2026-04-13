# -*- coding: utf-8 -*-
from data_loader import *
from process import *
import torch
from tqdm import tqdm

import shutup

shutup.please()

# 1 准备参数
conf = Config()
word2id, id2word = get_vocab(conf.train_data_path)

vocab_size = len(word2id)
pos_size = 143
tag_size = len(rel2id)
id2relation = {int(value): key for key, value in rel2id.items()}

# 2 准备数据、模型
_, test_iter = get_loader()
ba_model = BiLSTM_Attn(conf, vocab_size, pos_size, tag_size).to(conf.device)
ba_model.load_state_dict(torch.load(r'D:\workspace\code\projects\ai\knowledge_graph\relation_extraction\re_lstm_attention\save\best.pth',
                                    map_location=conf.device))


# 3 开始预测
# def model2predict():
#     ba_model.eval()
#     with torch.no_grad():
#         for sent, pos1, pos2, label, original_sequences, original_labels, entites in tqdm(test_iter):
#             sent = sent.to(conf.device)
#             pos1 = pos1.to(conf.device)
#             pos2 = pos2.to(conf.device)
#
#             print(f'original_sequences--->{original_sequences}')
#             print(f'original_labels--->{original_labels}')
#             print(f'entites--->{entites}')
#
#             output = ba_model(sent, pos1, pos2)
#             preds = torch.argmax(output, dim=1).tolist()
#             print(f'preds--->{preds}')
#
#             for i in range(len(original_sequences)):
#                 original_sequence = ''.join(original_sequences[i])
#                 original_label = id2relation[original_labels[i]]
#                 entity = entites[i]
#                 predict_label = id2relation[preds[i]]
#                 print('原始句子: ', original_sequence)
#                 print('原始关系: ', original_label)
#                 print('实体列表：', entity)
#                 print('模型预测: ', predict_label)
#                 print('*' * 80)
#                 # exit()


def model2predict():
    ba_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sent, pos1, pos2, label, original_sequences, original_labels, entites in tqdm(test_iter):
            sent = sent.to(conf.device)
            pos1 = pos1.to(conf.device)
            pos2 = pos2.to(conf.device)

            output = ba_model(sent, pos1, pos2)
            preds = torch.argmax(output, dim=1).tolist()

            # 统计正确率
            for pred, true_label in zip(preds, original_labels):
                if pred == true_label:
                    correct += 1
                total += 1

        accuracy = correct / total
        print(f'\n总样本数：{total}')
        print(f'正确预测：{correct}')
        print(f'准确率：{accuracy:.4f}')
        print(f'错误率：{1 - accuracy:.4f}')


if __name__ == '__main__':
    model2predict()
