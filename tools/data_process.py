# coding: utf-8
import json

with open('./train.txt', 'w', encoding='utf-8') as fw:
    for line in open('./data.json', 'r', encoding='utf-8'):
        data = json.loads(line)
        text = data['text']
        label = data['label']

        res = {}
        for one in label:
            start = int(one[0])
            end = int(one[1])
            cate = one[2]
            for i in range(start, end):
                if i == start:
                    tag = 'B-' + cate
                else:
                    tag = 'I-' + cate
                res[i] = tag

        for idx, char in enumerate(text):
            char_tag = res.get(idx, 'O')
            fw.write(char + '\t' + char_tag + '\n')
    fw.flush()
