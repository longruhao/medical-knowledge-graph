import json
import os

# 项目所在路径 增强代码健壮性
os.chdir('../..')
cur = os.getcwd()
print('项目路径：', cur)


class TransferData():
    '''
    转换样本格式，符合模型训练：序列标准任务，token 级别的分类
    '''

    def __init__(self):  # 一、加载数据
        # 1 实体类别 中英对照字典 label_dict
        self.label_dict = json.load(open(os.path.join(cur, 'LSTM_CRF/data/labels.json')))
        # 2 实体 标签-id 字典 tag2id_dict
        self.tag2id_dict = json.load(open(os.path.join(cur, 'LSTM_CRF/data/tag2id.json')))
        # 3 原始数据路径 origin_path
        self.origin_path = os.path.join(cur, 'LSTM_CRF/data_origin')
        # 4 样本格式转换之后 数据集保存路径 train_file_path
        self.train_file_path = os.path.join(cur, 'LSTM_CRF/data/train.txt')

    def transfer(self):  # 二、执行转换样本格式
        # 1：获取 原始文本文件路径(.txtoriginal.txt)、标签文件路径(.txt)
        # 2：写入 token 及其 标签，写入 self.train_file_path
        # 1.1 创建并打开 train.txt, 用于保存转换之后的数据
        with open(self.train_file_path, 'w', encoding='utf-8') as fw:
            # 1.2 逐级循环遍历文件
            # root 路径、dirs 路径下的文件夹列表、files 路径下的文件列表
            for root, dirs, files in os.walk(self.origin_path):
                # 1.3 循环遍历 当前级 路径下的 文件列表
                for file in files:
                    # 保证 file 是 原始文本文件(.txtoriginal.txt)
                    if 'original' not in file:
                        continue
                    # 1.4 拼接 原始文本文件路径 file_path、标签文件路径 label_file_path
                    file_path = os.path.join(root, file)
                    label_file_path = file_path.replace('.txtoriginal', '')
                    # 2.1 获取 {token索引：token标签} 字典 res_dict
                    # 形如 {4: 'B-LOC', 5: 'I-LOC', 6: 'I-LOC'}-'我爱美丽内蒙古'-实体'内蒙古'
                    res_dict = self.read_label_text(label_file_path)
                    # 2.2 读取原始文本，拿到非实体，结合 res_dict，写入 token 和标签
                    # res_dict 只有实体的标签，没有非实体，非实体来自原始文本
                    with open(file_path, 'r', encoding='utf-8') as fr:
                        # 一次拿到整个原始文本句子
                        content = fr.read().strip()
                        # 遍历原始文本中的每个字符char，并获取标签
                        for idx, char in enumerate(content):
                            # 获取标签 char_label，通过字典get方法
                            char_label = res_dict.get(idx, 'O')
                            # 写入字符和标签
                            fw.write(char + '\t' + char_label + '\n')

    def read_label_text(self, label_file_path):  # 三、获取 {实体token索引：实体token标签} 字典
        # 1：获得一个样本，分割成需要的元素
        # 2：组装 {token索引：token标签} 字典
        # 1.1 初始化要返回的字典 res_dict
        res_dict = {}
        # 1.2 打开标签文件
        with open(label_file_path, 'r', encoding='utf-8') as fr:
            # 1.3 逐行遍历样本，分割成需要的元素
            # line --> '右髋部\t21\t23\t身体部位'
            for line in fr.readlines():
                # 切割为 res ['右髋部', '21', '23', '身体部位']
                res = line.strip().split('\t')
                # 获取实体的 开始start、结束end 索引
                start = int(res[1])
                end = int(res[2])
                # 获取中文实体类别label
                label = res[3]
                # 获取英文实体类别label_tag
                label_tag = self.label_dict[label]
                # 2.1 遍历索引 (start, end+1) 组装标签tag 形如'B-LOC'，并保存到res_dict
                for i in range(start, end + 1):
                    if i == start:
                        tag = 'B-' + label_tag
                    else:
                        tag = 'I-' + label_tag
                    # 将 i:tag 添加到字典中
                    res_dict[i] = tag
        # 2.2 返回 {token索引：token标签} 字典
        return res_dict


if __name__ == '__main__':
    td = TransferData()
    td.transfer()
