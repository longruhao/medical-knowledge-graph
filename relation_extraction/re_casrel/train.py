from model.CasrelModel import *
from utils.process import *
from utils.data_loader import *
from config import *
import pandas as pd
from tqdm import tqdm
import shutup

shutup.please()


def model2train(model, train_iter, dev_iter, optimizer, conf):
    epochs = conf.epochs
    best_triple_f1 = 0  # 初始化最优f1值
    for epoch in range(epochs):
        train_epoch(model, train_iter, dev_iter, optimizer, best_triple_f1, epoch)
    # 保存最后一步的模型
    torch.save(model.state_dict(), '../save_model/last_model.pth')


# 训练内部迭代函数
def train_epoch(model, train_iter, dev_iter, optimizer, best_triple_f1, epoch):
    for step, (inputs, labels) in enumerate(tqdm(train_iter)):
        logist = model(**inputs)
        my_loss = model.compute_loss(**logist, **labels)
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()

        # 打印日志 保存模型
        if (step % 500 == 0) and (step != 0):
            results = model2dev(model, dev_iter)
            # results[-2] 是当前 step 的 triple_f1
            if results[-2] > best_triple_f1:
                best_triple_f1 = results[-2]
                torch.save(model.state_dict(), '../save_model/best_f1.pth')
                print('epoch:{},'
                      'step:{},'
                      'sub_precision:{:.4f}, '
                      'sub_recall:{:.4f}, '
                      'sub_f1:{:.4f}, '
                      'triple_precision:{:.4f}, '
                      'triple_recall:{:.4f}, '
                      'triple_f1:{:.4f},'
                      'train_loss:{:.4f}'.format(epoch,
                                                 step,
                                                 results[0],
                                                 results[1],
                                                 results[2],
                                                 results[3],
                                                 results[4],
                                                 results[5],
                                                 my_loss.item()))

    # 返回当前 epoch 的最佳 f1
    return best_triple_f1


# 验证函数
def model2dev(model, dev_iter):
    # print('进入验证循环')
    model.eval()
    # 定义一个df，来展示模型指标
    df = pd.DataFrame(columns=['TP', 'PRED', "REAL", 'p', 'r', 'f1'], index=['sub', 'triple'])
    df.fillna(0, inplace=True)
    for inputs, labels in tqdm(dev_iter):
        logits = model(**inputs)
        # 将预测值转化为 0 和 1
        pred_sub_heads = convert_score_to_zero_one(logits['pred_sub_heads'])
        pred_sub_tails = convert_score_to_zero_one(logits['pred_sub_tails'])
        pred_obj_heads = convert_score_to_zero_one(logits['pred_obj_heads'])
        pred_obj_tails = convert_score_to_zero_one(logits['pred_obj_tails'])

        # 标签值本身就是 0 和 1
        sub_heads = convert_score_to_zero_one(labels['sub_heads'])
        sub_tails = convert_score_to_zero_one(labels['sub_tails'])
        obj_heads = convert_score_to_zero_one(labels['obj_heads'])
        obj_tails = convert_score_to_zero_one(labels['obj_tails'])

        batch_size = inputs['input_ids'].shape[0]

        # pred_sub_heads.shape = [8, 70, 1]
        # pred_sub_heads[batch_index].shape = [70, 1]
        # pred_sub_heads[batch_index].squeeze().shape = [70]
        # pred_obj_heads.shape = [8, 70, 18]
        # pred_obj_heads[batch_index].shape = [70, 18]
        for batch_index in range(batch_size):
            # 提取 主体
            pred_subs = extract_sub(pred_sub_heads[batch_index].squeeze(1), pred_sub_tails[batch_index].squeeze(1))
            true_subs = extract_sub(sub_heads[batch_index].squeeze(), sub_tails[batch_index].squeeze())
            # 提取 客体-关系
            pred_ojbs = extract_obj_and_rel(pred_obj_heads[batch_index], pred_obj_tails[batch_index])
            true_objs = extract_obj_and_rel(obj_heads[batch_index], obj_tails[batch_index])

            # 填充 df
            df['PRED']['sub'] += len(pred_subs)
            df['REAL']['sub'] += len(true_subs)

            for true_sub in true_subs:
                if true_sub in pred_subs:
                    df['TP']['sub'] += 1

            df['PRED']['triple'] += len(pred_ojbs)
            df['REAL']['triple'] += len(true_objs)
            for true_obj in true_objs:
                if true_obj in pred_ojbs:
                    df['TP']['triple'] += 1

    # 计算 sub 指标，df[][] 先列后行
    sub_precision = df['TP']['sub'] / (df['PRED']['sub'] + 1e-9)
    sub_recall = df['TP']['sub'] / (df['REAL']['sub'] + 1e-9)
    sub_f1 = 2 * sub_precision * sub_recall / (sub_precision + sub_recall + 1e-9)

    df['p']['sub'] = sub_precision
    df['r']['sub'] = sub_recall
    df['f1']['sub'] = sub_f1

    # 计算 triple 指标
    triple_precision = df['TP']['triple'] / (df['PRED']['triple'] + 1e-9)
    triple_recall = df['TP']['triple'] / (df['REAL']['triple'] + 1e-9)
    triple_f1 = 2 * triple_precision * triple_recall / (
            triple_precision + triple_recall + 1e-9)

    df['p']['triple'] = triple_precision
    df['r']['triple'] = triple_recall
    df['f1']['triple'] = triple_f1

    return sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df


if __name__ == '__main__':
    conf = Config()
    model, optimizer, sheduler, device = load_model(conf)
    train_iter, dev_iter, test_iter = get_data()
    # 模型训练
    model2train(model, train_iter, dev_iter, optimizer, conf)
