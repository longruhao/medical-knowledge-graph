# 1 导包
import jieba.posseg as pseg
import re

# 2 设计关键字知识库 org_tag 保存实体结尾
org_tag = ['总局', '公司', '有限公司']


# 3 定义实体抽取函数 extract_org
def extract_org(text):
    # 3.1 使用词性标注分词 拿到分词结果 words_flags
    words_flags = pseg.lcut(text)
    # print(words_flags)

    # 3.2 循环遍历分词结果，分别保存 token 和 标签 到 words features
    # 结尾标记为 E、地址标记为 S、其他为 O
    words, features = [], []
    for word, flag in words_flags:
        words.append(word)
        if word in org_tag:
            features.append('E')
        else:
            if flag == 'ns':
                features.append('S')
            else:
                features.append('O')

    # 3.3 组合标签，拿到实体
    res = []
    features = ''.join(features)
    pattern = re.compile('S+O*E+')
    ner_label = re.finditer(pattern, features)
    # print('ner_label:',list(ner_label))

    for i in ner_label:
        start = i.start()
        end = i.end()
        # print(words[start:end])
        one_ner = ''.join(words[start:end])
        res.append(one_ner)

    return res


# 4 函数测试
text = '李连杰壹基金和韩红爱心慈善基金会向甘肃积石山县人民政府捐献了人民币两千万，帮助灾后重建'
text = '北京强盛科技股份有限公司向四川雅安人民政府，捐献人民币两亿元整，为了救助地震中受伤的孩子'
text = "可在接到本决定书之日起六十日内向中国国家市场监督管理总局申请行政复议,杭州海康威视数字技术股份有限公司."
print(extract_org(text))
