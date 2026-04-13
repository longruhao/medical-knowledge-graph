import json
from tqdm import tqdm
from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher
from predict import *
from config import *

conf = Config()

# 连接neo4j数据库，输入地址、用户名、密码、数据库名
# 7474是http协议端口号，7687是bolt协议端口号
graph = Graph('http://127.0.0.1:7474/', auth=('neo4j', '1234qwer'), name='hhh')

graph.match_one()
# 实例化节点查询对象
node_match = NodeMatcher(graph)
# 实例化关系查询对象
relation_match = RelationshipMatcher(graph)


def get_spo_type():
    ''''
    # 获取 每种关系类型 对应的 实体类型
    1、在正常使用模型预测spo时，因为模型不能给出subject和object的类型，所以借助训练集中的类型即可
    2、如果训练集也没有，可以根据预测关系的类别自己定义
    比如：出生日期，subject一般为人名，object一般为日期或时间
    '''
    spo_type = {}
    with open(conf.train_data, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            line = json.loads(line)
            spo_list = line['spo_list']
            for spo in spo_list:
                if spo['predicate'] not in spo_type:
                    spo_type[spo['predicate']] = spo
            if len(spo_type) == 18:  # 总共 18 种关系
                break
    # 保存到本地
    # with open('spo_type.json', 'a', encoding='utf-8') as fw:
    #     # 每层缩进4空格，方便阅读json
    #     fw.write(json.dumps(spo_type, ensure_ascii=False, indent=4))
    return spo_type


def ready_data():
    '''
    通过加载训练好的模型，对需要预测关系的文本进行预测
    并补充实体类型 subject_type object_type，将结果存储到文件 predict_spo.json
    '''
    # 获取 每种关系类型 对应的 实体类型
    type_dict = get_spo_type()
    # 导入训练好的模型
    model_path = '../save_model/last_model.pth'
    my_model = load_model(model_path)
    # 读取 测试集test.json 进行预测
    with open(conf.test_data, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    for line in tqdm(lines):
        line = json.loads(line)
        # print(line)
        sample = line['text']
        outputs = model2predict(sample, my_model)
        if len(outputs) == 0:
            continue
        spo_list = outputs['spo_list']
        # print(spo_list)
        for spo in spo_list:
            if spo['predicate'] in type_dict:
                spo['subject_type'] = type_dict[spo['predicate']]['subject_type']
                spo['object_type'] = type_dict[spo['predicate']]['object_type']
                with open('../data/predict_spo.json', 'a', encoding='utf-8') as fw:
                    # json.dumps() 将字典转化为 json 格式的字符串
                    # ensure_ascii=False：非 ASCII 字符将原样保留
                    fw.write(json.dumps(spo, ensure_ascii=False) + '\n')


# 创建节点
def create_node(graph, label, attrs):  # attrs 属性
    # 第一次：如果你的图数据库之前的数据是没有用的可以直接清空
    # graph.delete_all()
    # 构建节点的属性，如果一个节点有多个属性，所以加上and
    condition = ""
    for key, value in attrs.items():
        condition += '_.%s=' % key + '\"' + value + '\"' + " and "
    # 将condition最后的 and字符 去掉
    condition = condition[:-5]
    # 根据属性条件查询节点是否已经存在，若存在则返回该节点，否则返回None
    value = node_match.match(label).where(condition).first()
    # 如果要创建的节点不存在则再创建
    if value is None:
        node = Node(label, **attrs)
        node = graph.create(node)
        return node
    return None


# 创建关系
def create_relationship(graph, label1, attrs1, label2, attrs2, r_name):
    value1 = match_node(label1, attrs1)
    value2 = match_node(label2, attrs2)
    # 判断实体是否均存在，否则无法创建关系
    if value1 is None or value2 is None:
        return False
    # 判断是否已经创建完关系，如果已经创建就不用再重复定义了
    # 当然这一步也可以省略，因为相同实体对创建相同关系时，结果会覆盖
    # rel_value = match_relation(node1=value1, node2=value2, r_type=r_name)
    # print("rel_value:{}".format(rel_value))
    # if rel_value:
    #     return False
    r = Relationship(value1, r_name, value2)
    graph.create(r)


# 使用 NodeMatcher 查询节点
def match_node(label, attrs):
    condition = ""
    for key, value in attrs.items():
        condition += '_.%s=' % key + '\"' + value + '\"' + " and "
    # 将condition最后的 and 字符去掉(注意and前后有两个空格)
    condition = condition[:-5]
    # 根据属性条件查询节点是否已经存在，若存在则返回该节点，否则返回None
    # 加上.first()返回一个节点结果，不加.first()返回所有符合要求的节点结果
    value = node_match.match(label).where(condition).first()
    return value


# 使用 RelationshipMatcher 查询关系
def match_relation(node1, node2, r_type):
    relationship_list = list(relation_match.match((node1, node2), r_type=r_type))
    if len(relationship_list) == 0:
        return False
    else:
        return True


# 读取文件，创建图谱
def load_file_create_map():
    # 首先需要运行 ready_data()函数 获取模型预测的 spo 数据，通过文件形式存储数据，方便反复使用
    # 首次运行可以清空所有图数据
    graph.delete_all()
    # 第一步获取数据
    with open('../data/predict_spo.json', 'r', encoding='utf-8') as fr:
        for line in tqdm(fr.readlines()):
            line = json.loads(line)
            # todo: 定义主实体节点
            # 定义 主实体 节点的标签
            subject_label = line["subject_type"]
            # 定义 主实体 节点的属性
            sub_attrs = {'name': line["subject"]}
            create_node(graph, subject_label, sub_attrs)

            # todo: 定义客实体节点
            # 定义 客实体 节点的标签
            object_label = line["object_type"]
            # 定义 客实体 节点的属性
            if "日期" in line['predicate']:
                obj_attrs = {'date': line["object"]}
            else:
                obj_attrs = {'name': line["object"]}
            create_node(graph, object_label, obj_attrs)

            # todo: 定义主、客实体的关系
            r_name = line["predicate"]
            create_relationship(graph, subject_label, sub_attrs, object_label, obj_attrs, r_name)


# 查询 neo4j 数据
def use_neo4j2search():
    # 查询图中 节点 和 关系类型 有哪些
    node_labels = graph.schema.node_labels  # 查询一共有多少种节点类型
    print(F"Neo4j图数据库中存在的节点类型为：{node_labels}")
    relation_types = graph.schema.relationship_types  # 查询一共有多少种关系类型
    print(F"Neo4j图数据库中存在的关系类型为：{relation_types}")
    # 查询节点，这里以label="人物", attrs = {"name": "李晨"} 为例
    node1 = match_node(label='人物', attrs={"name": '李晨'})
    print(node1)
    # 查询关系，以 “李晨”节点 为出发点，进行关系的查询
    # eg1：查询 “李晨”节点 的所有关系：先查节点，再查关系，r_type=None表示任意类型的关系
    ship_list1 = list(relation_match.match([node1], r_type=None))
    for reship in ship_list1:
        print(reship)
    print('*' * 80)

    # eg2：查询 “李晨” 和 “中国” 的关系：两个节点的顺序表示要匹配的关系方向
    node2 = match_node(label='国家', attrs={"name": '中国'})
    ship_list12 = list(relation_match.match([node1, node2], r_type=None))
    for reship in ship_list12:
        print(reship)
    print('*' * 80)

    # eg3：查询某一类关系：第一个参数为None，第二个参数r_type=指定关系类型
    ship_list3 = list(relation_match.match(None, r_type="出生日期"))
    for reship in ship_list3:
        print(reship)
        break


if __name__ == '__main__':
    # load_file_create_map()
    use_neo4j2search()
