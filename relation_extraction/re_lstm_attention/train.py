# -*- coding: utf-8 -*-
import time
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import torch.optim as optim
from data_loader import *

import shutup

shutup.please()


def train(conf, vocab_size, pos_size, tag_size):
    # 1 准备物料
    train_loader, test_loader = get_loader()
    model = BiLSTM_Attn(conf, vocab_size, pos_size, tag_size).to(conf.device)
    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # 2 初始化几个参数
    # 累计损失train_loss、累计预测正确的样本数train_acc
    # 累计步数total_step、累计样本数total_sample
    start_time = time.time()
    total_loss = 0
    total_acc = 0
    total_step = 0
    total_sample = 0
    f1_init = -1000

    # 3 开启训练
    for epoch in range(conf.epochs):
        model.train()
        for idx, (sent, pos1, pos2, label, _, _, _) in enumerate(tqdm(train_loader, desc="Model Training")):
            # 3.1 训练过程
            sent = sent.to(conf.device)
            pos1 = pos1.to(conf.device)
            pos2 = pos2.to(conf.device)
            label = label.to(conf.device)
            output = model(sent, pos1, pos2)
            loss = loss_fn(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 3.2 累计统计
            # 累计损失
            total_loss += loss.item()
            # 累计预测正确的样本数
            total_acc += sum(torch.argmax(output, dim=1) == label).item()
            # 累计步数
            total_step += 1
            # 累计样本数
            total_sample += len(label)

            # 3.3 打印日志
            if idx % (len(train_loader) // 2) == 0 and idx:
                aver_loss = total_loss / total_step
                aver_acc = total_acc / total_sample
                print('epoch：%d, loss:%.3f, acc:%.3f' % (epoch, aver_loss, aver_acc))

        # 3.4 根据模型测试结果 保存模型
        precision, recall, f1, report, loss = model2test(test_loader, model, loss_fn)
        if f1 > f1_init:
            f1_init = f1
            torch.save(model.state_dict(), 'D:\workspace\code\projects\ai\knowledge_graph\relation_extraction\re_lstm_attention\save\best.pth')
            print('Test Loss:%.4f\t' % loss)
            print(report)

    # 3.5 保存最后一个轮次的model，打印耗时
    torch.save(model.state_dict(), 'D:\workspace\code\projects\ai\knowledge_graph\relation_extraction\re_lstm_attention\save\last.pth')
    print('训练总耗时：', time.time() - start_time)


def model2test(test_loader, model, loss_fn):
    golds = []
    preds = []
    test_loss = 0
    model.eval()
    for sent, pos1, pos2, label, _, _, _ in tqdm(test_loader, desc="Model Testing"):
        sent = sent.to(conf.device)
        pos1 = pos1.to(conf.device)
        pos2 = pos2.to(conf.device)
        label = label.to(conf.device)
        output = model(sent, pos1, pos2)
        loss = loss_fn(output, label)
        test_loss += loss.item()
        output = torch.argmax(output, dim=-1)
        preds.extend(output.tolist())
        golds.extend(label.tolist())

    loss = test_loss / len(test_loader)
    precision = precision_score(golds, preds, average='macro')
    recall = recall_score(golds, preds, average='macro')
    f1 = f1_score(golds, preds, average='macro')
    report = classification_report(golds, preds)
    return precision, recall, f1, report, loss


if __name__ == '__main__':
    word2id, id2word = get_vocab(conf.train_data_path)
    vocab_size = len(word2id)
    print(vocab_size)
    pos_size = 142
    tag_size = len(rel2id)
    train(conf, vocab_size, pos_size, tag_size)
