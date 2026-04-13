from config import *
from model.BiLSTM import *
from model.BiLSTM_CRF import *
from utils.data_loader import *
import time
from tqdm import tqdm
# classification_report可以导出字典格式，修改参数：output_dict=True，可以将字典在保存为csv格式输出
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import torch.optim as optim
import shutup

shutup.please()
conf = Config()


def model2train():
    # 1 准备物料 训练和验证的dataloader、model、loss_fn、optimizer
    train_loader, dev_loader = get_data()
    models = {'BiLSTM': NERLSTM, 'BiLSTM_CRF': NERLSTM_CRF}
    model = models[conf.model](conf.embedding_dim, conf.hidden_dim, conf.dropout, word2id, conf.tag2id)
    model = model.to(conf.device)
    loss_fn = nn.CrossEntropyLoss()  # for BiLSTM
    optimizer = optim.AdamW(model.parameters(), lr=conf.lr)

    # 2 开始训练 设置初试时间、f1
    start_time = time.time()
    f1_score = -1000
    # 模型不同 训练不同
    if conf.model == 'BiLSTM':
        # 2.1 循环轮次，开启训练
        for epoch in range(conf.epochs):
            model.train()
            # 2.2 逐 step 训练，数据放在设备上，每隔200打印观测值
            for idx, (inputs, labels, mask) in enumerate(tqdm(train_loader, desc='BiLSTM Training')):
                x = inputs.to(conf.device)
                tags = labels.to(conf.device)
                mask = mask.to(conf.device)
                pred = model(x, mask)
                # 预测值是三维的，loss_fn不支持，需要view(-1, len(conf.tag2id))
                # 同时 tags.view(-1)
                pred = pred.view(-1, len(conf.tag2id))
                loss = loss_fn(pred, tags.view(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if idx % 50 == 0:
                    print('Epoch:%d\t\tLoss:%.4f' % (epoch, loss.item()))
            # 2.3 每轮验证 1 次，根据 f1 保存模型
            precision, recall, f1, report = model2dev(dev_loader, model, loss_fn)
            if f1 > f1_score:
                f1_score = f1
                torch.save(model.state_dict(), 'save_model/bilstm_best.pth')
                print(report)
        # 2.4 训练结束，打印耗时
        end_time = time.time()
        print(f'训练总耗时：{end_time - start_time}')

    elif conf.model == 'BiLSTM_CRF':
        # 2.1 循环轮次，开启训练
        for epoch in range(conf.epochs):
            model.train()
            # 2.2 逐 step 训练，数据放在设备上，每隔一定时间打印观测值
            for idx, (inputs, labels, mask) in enumerate(tqdm(train_loader, desc='BiLSTM_CRF Training')):
                x = inputs.to(conf.device)
                tags = labels.to(conf.device)
                mask = mask.to(conf.device)
                loss = model.log_likelihood(x, tags, mask)
                loss.backward()
                # 梯度裁剪 防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
                optimizer.step()
                optimizer.zero_grad()
                if idx % 50 == 0:
                    print('Epoch:%d\tBatch:%d\tLoss:%.4f' % (epoch, idx, loss.item()))
            # 2.3 每轮验证 1 次，根据 f1 保存模型
            precision, recall, f1, report = model2dev(dev_loader, model, loss_fn)
            if f1 > f1_score:
                f1_score = f1
                torch.save(model.state_dict(), 'save_model/bilstm_crf_best.pth')
                print(report)
        # 2.4 训练结束，打印耗时
        end_time = time.time()
        print(f'训练总计耗时：{end_time - start_time}')


def model2dev(dev_loader, model, loss_fn=None):  # 模型验证
    # 1 初始化 平均损失aver_loss、总的预测结果preds、总的真实标签golds，并开启验证
    aver_loss = 0
    preds, golds = [], []
    model.eval()

    # 2 逐 step 验证，数据放在设备上
    for idx, (inputs, labels, mask) in enumerate(tqdm(dev_loader, desc="Model Validation")):
        val_x = inputs.to(conf.device)
        val_y = labels.to(conf.device)
        mask = mask.to(conf.device)
        # 3 初始化 保存 batch 预测结果的列表 predict，分模型拿到预测值
        predict = []
        if model.name == "BiLSTM":
            pred = model(val_x, mask)
            predict = torch.argmax(pred, dim=-1).tolist()
            pred = pred.view(-1, len(conf.tag2id))
            val_loss = loss_fn(pred, val_y.view(-1))
            aver_loss += val_loss.item()
        elif model.name == "BiLSTM_CRF":
            predict = model(val_x, mask)  # crf 前向传播的结果本身就是 list
            loss = model.log_likelihood(val_x, val_y, mask)
            aver_loss += loss.item()

        # # 统计 真实标签，句子非 pad 的部分
        # leng = []
        # for i in val_y.cpu():
        #     tmp = []
        #     for j in i:
        #         if j.item() >= 0:
        #             tmp.append(j.item())
        #     leng.append(tmp)
        #
        # # 提取真实长度的预测标签
        # for index, i in enumerate(predict):
        #     preds.extend(i[:len(leng[index])])
        #
        # # 提取真实长度的真实标签
        # for index, i in enumerate(val_y.tolist()):
        #     golds.extend(i[:len(leng[index])])

        # 4 去掉 预测值 predict 和 val_y 中的 pad 部分，pad 用 11 表示
        for one_pred, one_true in zip(predict, val_y.tolist()):
            pad_len = one_true.count(11)
            no_pad_len = len(one_true) - pad_len
            preds.extend(one_pred[:no_pad_len])
            golds.extend(one_true[:no_pad_len])

    # 5 计算评估指标
    aver_loss /= (len(dev_loader) * conf.batch_size)
    precision = precision_score(golds, preds, average='macro')  # macro 宏平均，平等看待所有类别
    recall = recall_score(golds, preds, average='macro')
    f1 = f1_score(golds, preds, average='macro')
    report = classification_report(golds, preds)
    return precision, recall, f1, report


if __name__ == '__main__':
    model2train()
