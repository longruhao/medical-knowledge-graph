from config import *

conf = Config()


def build_data():  # 构造数据集
    # 1 初始化容器
    # 总样本datas、一个句子的sample_x sample_y
    # 词表vocab_list = ["PAD", 'UNK']
    datas = []
    sample_x = []
    sample_y = []
    vocab_list = ["PAD", 'UNK']
    # 2 遍历训练集，将句子分段，保存在 datas
    # 逐行读取 train.txt，line --> '咳\tB-SIGNS'
    # for line in open() 这种方式：处理大文件时，逐行读取会节省内存
    for line in open(conf.train_path, 'r', encoding='utf-8'):
        # 2.1 切割 line，并获得 ['咳','B-SIGNS']，添加到 sample_x、sample_y
        line = line.strip().split('\t')
        # 如果 len(line) != 2 直接跳过
        if len(line) != 2:
            continue
        char = line[0]
        cate = line[-1]
        sample_x.append(char)
        sample_y.append(cate)
        # 2.2 填充词表
        if char not in vocab_list:
            vocab_list.append(char)
        # 2.3 截断句子，如果 char 在 ['?', '!', '。', '？', '！']，截断当前
        # 作为一个独立句子，把 [sample_x, sample_y] append 到 datas
        # 并清空容器 sample_x、sample_y
        if char in ['?', '!', '。', '？', '！']:
            datas.append([sample_x, sample_y])
            sample_x = []
            sample_y = []
    # 3 构建 word2id，enumerate(vocab_list)
    word2id = {word: idx for idx, word in enumerate(vocab_list)}
    # 4 保存词表文件 vocab.txt
    write_file(vocab_list, conf.vocab_path)
    return datas, word2id


def write_file(vocab_list, file_path):  # 保存词表文件 vocab.txt
    # 换行 拼接 写入 '\n'.join(vocab_list)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab_list))


if __name__ == '__main__':
    datas, word2id = build_data()
    print(len(datas))
    print(datas[:2])
    print(word2id)
    print(len(word2id))
