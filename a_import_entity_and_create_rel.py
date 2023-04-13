import os.path
import random
import re
import py2neo
from tqdm import tqdm

cipin = {
        "疾病": {},
        "药品": {},
        "食物": {},
        "检查项目": {},
        "科目": {},
        "疾病症状": {},
        "治疗方法": {},
        "药品商": {},
    }

def create_node(client,type,name):
    # node = py2neo.Node(type,名称=name)
    # client.create(node)
    order = """create (n:%s{名称:"%s"})"""%(type,name)
    client.run(order)
#导入普通实体
def import_entity(client,type,entity):
    print(f'正在导入{type}类数据')
    for en in tqdm(entity):
        create_node(client,type,en)
#导入疾病类实体
def import_disease_data(client,type,entity):
    print(f'正在导入{type}类数据')
    for disease in tqdm(entity):
        node = py2neo.Node(type,
                           名称=disease["名称"],
                           疾病简介=disease["疾病简介"],
                           疾病病因=disease["疾病病因"],
                           预防措施=disease["预防措施"],
                           治疗周期=disease["治疗周期"],
                           治愈概率=disease["治愈概率"],
                           疾病易感人群=disease["疾病易感人群"],

                           )
        client.create(node)
def create_relationship(client,type1, name1,relation, type2,name2):
    order = """match (a:%s{名称:"%s"}),(b:%s{名称:"%s"}) create (a)-[r:%s]->(b)"""%(type1,name1,type2,name2,relation)
    client.run(order)
def create_all_relationship(client,all_relationship):
    print("正在导入关系.....")
    for type1, name1,relation, type2,name2  in tqdm(all_relationship):
        create_relationship(client,type1, name1,relation, type2,name2)
import ahocorasick
class rule_find:
    def __init__(self,all_entity):
        self.ahos = [ahocorasick.Automaton() for i in range(len(all_entity))]
        self.idx2type = list(all_entity.keys())
        self.type2idx = {k:v for v,k in enumerate(self.idx2type)}
        for type,entities in all_entity.items():

            for en in entities:
                if len(en)>=2:
                    self.ahos[self.type2idx[type]].add_word(en,en)
        for i in range(len(self.ahos)):
            self.ahos[i].make_automaton()
    def find(self,sen):
        rule_result = []
        for i in range(len(self.ahos)):
            all_res = list(self.ahos[i].iter(sen))
            for res in all_res:
                if res[1] in  cipin[self.idx2type[i]]:
                    cipin[self.idx2type[i]][res[1]] +=1
                else:
                    cipin[self.idx2type[i]][res[1]] = 1
        return rule_result
if __name__ == "__main__":
    #连接neo4j库
    # client = py2neo.Graph('http://localhost:7474', user='neo4j', password='wei8kang7.long', name='neo4j')

    # is_delete = input('注意:是否删除neo4j上的所有实体 y/n')
    # if is_delete=='y':
    #     client.run("match (n) detach delete (n)")

    with open('./data/medical.json','r',encoding='utf-8') as f:
        all_data = f.read().split('\n')
    #所有实体
    all_entity = {
        "疾病": [],
        "药品": [],
        "食物": [],
        "检查项目":[],
        "科目":[],
        "疾病症状":[],
        "治疗方法":[],
        "药品商":[],
    }

    # 实体间的关系
    relationship = []






    #将所有实体导入到all_entity中，将所有的关系放到relationship中
    for data in all_data:
        if (len(data) < 3):
            continue
        data = eval(data)

        disease_name = data.get("name","")
        all_entity["疾病"].append({
            "名称":disease_name,
            "疾病简介": data.get("desc", ""),
            "疾病病因": data.get("cause", ""),
            "预防措施": data.get("prevent", ""),
            "治疗周期":data.get("cure_lasttime",""),
            "治愈概率": data.get("cured_prob", ""),
            "疾病易感人群": data.get("easy_get", ""),
        })

        drugs = data.get("common_drug", []) + data.get("recommand_drug", [])
        all_entity["药品"].extend(drugs)  # 添加药品实体
        if drugs:
            relationship.extend([("疾病", disease_name, "疾病使用药品", "药品",durg)for durg in drugs])

        do_eat = data.get("do_eat",[])+data.get("recommand_eat",[])
        no_eat = data.get("not_eat",[])
        all_entity["食物"].extend(do_eat+no_eat)
        if do_eat:
            relationship.extend([("疾病", disease_name,"疾病宜吃食物","食物",f) for f in do_eat])
        if no_eat:
            relationship.extend([("疾病", disease_name, "疾病忌吃食物", "食物", f) for f in no_eat])

        check = data.get("check", [])
        all_entity["检查项目"].extend(check)
        if check:
            relationship.extend([("疾病", disease_name, "疾病所需检查", "检查项目",ch) for ch in check])

        cure_department=data.get("cure_department", [])
        all_entity["科目"].extend(cure_department)
        if cure_department:
            relationship.append(("疾病", disease_name, "疾病所属科目", "科目",cure_department[-1]))

        symptom = data.get("symptom",[])
        all_entity["疾病症状"].extend(symptom)
        if symptom:
            relationship.extend([("疾病", disease_name, "疾病的症状", "疾病症状",sy )for sy in symptom])

        cure_way = data.get("cure_way", [])
        all_entity["治疗方法"].extend(cure_way)
        if cure_way:
            relationship.extend([("疾病", disease_name, "治疗的方法", "治疗方法", cure_w) for cure_w in cure_way])

        acompany_with = data.get("acompany", [])
        if acompany_with:
            relationship.extend([("疾病", disease_name, "疾病并发疾病", "疾病", disease) for disease in acompany_with])


        drug_detail = data.get("drug_detail",[])
        pattern = r'(.*?)\((.*?)\)'
        for detail in drug_detail:
            lis = re.findall(pattern,detail)
            if len(lis)!=0:
                d,p = lis[0]
                d = d.strip(p)
                all_entity["药品商"].append(d)
                all_entity["药品"].append(p)
                if d:
                    relationship.append(('药品商',d,"生产","药品",p))








    relationship = list(set(relationship))




    # # 保存关系 放到data下
    # with open("./data/rel.txt",'w',encoding='utf-8') as f:
    #     for rel in relationship:
    #         f.write(" ".join(rel))
    #         f.write('\n')




    # #将属性和实体导入到neo4j上,注:只有疾病有属性，特判
    # for k in all_entity:
    #     if k!="疾病":
    #         import_entity(client,k,all_entity[k])
    #     else:
    #         import_disease_data(client,k,all_entity[k])
    # create_all_relationship(client,relationship)



    #将读取的实体保存成文件，放到data中
    now_entity = all_entity.copy()['疾病']
    all_entity['疾病'] = [en['名称'] for en in all_entity['疾病']]
    all_entity = {k: list(set(v))  for k, v in all_entity.items()}
    rule = rule_find(all_entity)

    #初始化词频
    for ty,entities in all_entity.items():
        for en in entities:
            cipin[ty][en] = 0

    for des in tqdm(now_entity):
        rule.find(des['疾病简介'])
        rule.find(des['疾病病因'])
        rule.find(des['预防措施'])

    for name, entity in cipin.items():
        with open(os.path.join('data', 'ent1', f'{name}.txt'), 'w', encoding='utf-8') as f:
            for en, num in entity.items():
                f.write(f"{en} {num}\n")
    # for name,entity in all_entity.items():
    #     with open(os.path.join('data','ent1',f"{name}.txt"),'w',encoding='utf-8') as f:
    #         entity = [en.strip('，') for en in entity if (len(en)<=15 or random.random()<0.2) and len(en)>=2]
    #         f.write("\n".join(entity))





