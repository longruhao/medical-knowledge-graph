#-*- coding: utf-8 -*-
from model.CasrelModel import *
from utils.process import *
from utils.data_loader import *
from config import *
import pandas as pd
from tqdm import tqdm
import torch


def model2test(model, test_iter):
    '''
    测试模型效果
    :param model:
    :param test_iter:
    :return:
    '''
    model.eval()
    # 定义一个df，来展示模型的指标。
    df = pd.DataFrame(columns=['TP', 'PRED', "REAL", 'p', 'r', 'f1'], index=['sub', 'triple'])
    df.fillna(0, inplace=True)
    with torch.no_grad():
        for inputs, labels in tqdm(test_iter):
            logist = model(**inputs)
            pred_sub_heads = convert_score_to_zero_one(logist['pred_sub_heads'])
            pred_sub_tails = convert_score_to_zero_one(logist['pred_sub_tails'])
            sub_heads = convert_score_to_zero_one(labels['sub_heads'])
            sub_tails = convert_score_to_zero_one(labels['sub_tails'])
            batch_size = inputs['input_ids'].shape[0]
            obj_heads = convert_score_to_zero_one(labels['obj_heads'])
            obj_tails = convert_score_to_zero_one(labels['obj_tails'])
            pred_obj_heads = convert_score_to_zero_one(logist['pred_obj_heads'])
            pred_obj_tails = convert_score_to_zero_one(logist['pred_obj_tails'])

            for batch_index in range(batch_size):

                pred_subs = extract_sub(pred_sub_heads[batch_index].squeeze(), pred_sub_tails[batch_index].squeeze())
                true_subs = extract_sub(sub_heads[batch_index].squeeze(), sub_tails[batch_index].squeeze())

                pred_ojbs = extract_obj_and_rel(pred_obj_heads[batch_index], pred_obj_tails[batch_index])
                true_objs = extract_obj_and_rel(obj_heads[batch_index], obj_tails[batch_index])

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
        df.loc['sub', 'p'] = df['TP']['sub'] / (df['PRED']['sub'] + 1e-9)
        df.loc['sub', 'r'] = df['TP']['sub'] / (df['REAL']['sub'] + 1e-9)
        df.loc['sub', 'f1'] = 2 * df['p']['sub'] * df['r']['sub'] / (df['p']['sub'] + df['r']['sub'] + 1e-9)
        df.loc['triple', 'p'] = df['TP']['triple'] / (df['PRED']['triple'] + 1e-9)
        df.loc['triple', 'r'] = df['TP']['triple'] / (df['REAL']['triple'] + 1e-9)
        df.loc['triple', 'f1'] = 2 * df['p']['triple'] * df['r']['triple'] / (
                df['p']['triple'] + df['r']['triple'] + 1e-9)

    return df


if __name__ == '__main__':
    conf = Config()
    model, optimizer, sheduler, device = load_model(conf)
    model.load_state_dict(torch.load('../save_model/last_model.pth'))
    train_iter, dev_iter, test_iter = get_data()
    # 模型训练
    df = model2test(model, test_iter)
    print(df)
