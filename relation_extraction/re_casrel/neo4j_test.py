# coding: utf-8
from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher

graph = Graph("http://localhost:7474", auth=("neo4j", "1234qwer"), name='hhh')

# todo:第一次使用Neo4j图数据库可以清空一下
# graph.delete_all()
#
# ##todo:创建节点
# ##定义节点1，标签类别为Student, 属性：name:jack, age:18
# node1 = Node("Student", name="jack", age=18)  # 注意：这里不要加“label=”，否则label会以属性的形式展示，而非标签
# # # # 定义节点2，标签类别为Student, 属性：name:lucy, age:20
# node2 = Node("Student", name="lucy", age=20)
# # # 调用graph对象的create方法完成节点的创建
# # graph.create(node1)
# # graph.create(node2)
#
# ##todo:创建关系
# # # 实例化关系类对象
# relation1 = Relationship(node1, 'Friend', node2)
# # # 调用graph对象的create方法完成关系的创建
# graph.create(relation1)
# # # 实例化关系类对象
# relation2 = Relationship(node2, 'Friend', node1)
# # # 调用graph对象的create方法完成关系的创建
# graph.create(relation2)
# node1 = node_match.match('Student').where(name='jack').first()
# node3 = Node("Location", name="America")
# node4 = Node("Sports", name='football')
# relation3 = Relationship(node1, 'Country', node3)
# graph.create(relation3)
# relation4 = Relationship(node1, 'Hobby', node4)
# graph.create(relation4)
# # todo:查询节点和关系
##不带where条件查询节点
# nodes = list(node_match.match('Student'))
# print(nodes)
# node = node_match.match('Student').first()
# print(node)
# print('*'*80)
# ##带where条件查询:第一种用法
# nodes = list(node_match.match('Student').where(age=18))
# print(nodes)
# node = node_match.match('Student').where(age=18).first()
# print(node)
# print('*' * 80)
# ##带where条件查询:第二种用法
# nodes = list(node_match.match('Student').where("_.age>18"))
# print(nodes)
# node = node_match.match('Student').where("_.age>18").first()
# print(node)
# 查询关系
# 删除关系
# 查找“jack”节点对应的兴趣关系
# node1 = node_match.match('Student').where(name='lucy').first()
# print(node1)
# del node1['age']
# print(node1)
# graph.push(node1)
# node2 = node_match.match('Sports').where(name='football').first()
# print(node2)
# relationship = relation_match.match([node1, node2], r_type="Hobby").first()
# print(relationship)
# # print(relationship.identity)
# graph.delete(relationship)
# graph.separate(relationship)
# print(node1.identity)
# print(node2.identity)
# print(node_match[node2.identity])

cypher_ = "MATCH (n: Student) WHERE n.age > 16 RETURN n.name AS Name, n.age AS Age"
df = graph.run(cypher_).to_data_frame()
print(df)
labels = graph.schema.node_labels
print(labels)
relation_types = graph.schema.relationship_types
print(relation_types)
