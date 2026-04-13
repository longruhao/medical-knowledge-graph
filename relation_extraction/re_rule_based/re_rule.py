# 1 导包
import jieba.posseg as pseg

# 2 准备样本数据
samples = [
    "2014年1月8日，杨幂与刘恺威的婚礼在印度尼西亚巴厘岛举行",
    "周星驰和吴孟达在《逃学威龙》中合作出演",
    "成龙出演了《警察故事》等多部经典电影",
]

# 3 定义关系集合
rel2dict = {
    '夫妻关系': ['结婚', '领证', '婚礼'],
    '合作关系': ['搭档', '合作', '签约'],
    '演员关系': ['出演', '角色', '主演']
}

# 4 通过 jieba 抽取出 实体 和 关系词组，组建 spo 三元组，完成关系抽取
for text in samples:
    # 4.1 初始化 保存实体、关系的列表 ents、rels
    # jieba不能直接识别出电影名称，因此需要单独处理，初始化列表 movie_name
    ents = []
    rels = []
    movie_name = []

    # 4.2 循环遍历 词性标注结果
    for word, flag in pseg.lcut(text):
        # a 如果是人名，直接添加到 ents
        if flag == 'nr':
            ents.append(word)

        # b 如果是《》根据 len(movie_name) 判断第几个符号
        elif flag == 'x':
            if len(movie_name) == 0:
                movie_name.append(text.index(word))
            else:
                movie_name.append(text.index(word))
                ents.append(text[movie_name[0] + 1: movie_name[1]])

        # c 如果是关系
        else:
            for key, value in rel2dict.items():
                if word in value:
                    rels.append(key)

    # 4.3 判断长度是否符合，组合并返回
    if len(ents) >= 2 and len(rels) >= 1:
        print('原始文本：', text)
        print('提取结果：', (ents[0], rels[0], ents[1]))
    else:
        print('原始文本：', text)
        print('不好意思，没有找到结果')
