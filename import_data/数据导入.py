import py2neo
def create_node(client,type,name,sex,age):
    # node = py2neo.Node(type,姓名=name,性别=sex,年龄=age)
    # client.create(node)
    order = """create (n:%s{姓名:"%s",性别:"%s",年龄:%d})"""%(type,name,sex,age)
    client.run(order)
def create_relationship(client,type1,type2,name1,name2,relation):
    order = """match (a:%s{姓名:"%s"}),(b:%s{姓名:"%s"}) create (a)-[r:%s]->(b)"""%(type1,name1,type2,name2,relation)
    client.run(order)
def load_node(client,type):
    with open(f'{type}信息.txt','r',encoding='utf8') as f:
        all_data = f.read().split('\n')
    for data in all_data:
        data = data.strip().split(' ')
        if(len(data)!=3):
            continue
        name,sex,age = data
        create_node(client,type,name,sex,int(age))
def load_relationship(client,path):
    with open(path,'r',encoding='utf8') as f:
        all_data = f.read().split('\n')
    for data in all_data:
        data = data.strip().split(' ')
        if(len(data)!=5):
            continue
        type1,name1,relation,type2,name2 = data
        create_relationship(client,type1,type2,name1,name2,relation)
if __name__ == "__main__":
    client = py2neo.Graph('http://localhost:7474', user='neo4j',password='wei8kang7.long',name='neo4j')
    client.run("match (n) detach delete (n)")
    load_node(client,'学生')
    load_node(client, '老师')
    load_relationship(client, '总关系.txt')
    print()
