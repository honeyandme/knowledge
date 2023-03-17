import re
import py2neo
from tqdm import tqdm
def create_node(client,type,name):
    # node = py2neo.Node(type,名称=name)
    # client.create(node)
    order = """create (n:%s{名称:"%s"})"""%(type,name)
    client.run(order)
def import_entity(client,type,entity):
    print(f'正在导入{type}类数据')
    for en in tqdm(entity):
        create_node(client,type,en)
if __name__ == "__main__":
    client = py2neo.Graph('http://localhost:7474', user='neo4j', password='wei8kang7.long', name='neo4j')
    client.run("match (n) detach delete (n)")
    with open('medical.json','r',encoding='utf-8') as f:
        all_data = f.read().split('\n')
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
    for data in all_data:
        if (len(data) < 3):
            continue
        data = eval(data)
        all_entity["疾病"].append({
            "疾病名称":data.get("name",""),
            "疾病简介": data.get("desc", ""),
            "疾病病因": data.get("cause", ""),
            "预防措施": data.get("prevent", ""),
            "治疗周期":data.get("cure_lasttime",""),
            "治愈概率": data.get("cured_prob", ""),
            "疾病易感人群": data.get("easy_get", ""),
        })
        all_entity["药品"].extend(data.get("common_drug", []) + data.get("recommand_drug", []))  # 添加药品实体
        all_entity["食物"].extend(data.get("do_eat",[])+data.get("not_eat",[])+ data.get("recommand_eat",[]))
        all_entity["检查项目"].extend(data.get("check", []))
        all_entity["科目"].extend(data.get("cure_department", []))
        all_entity["疾病症状"].extend(data.get("symptom",[]))
        all_entity["治疗方法"].extend(data.get("cure_way", []))

        drug_detail = data.get("drug_detail",[])
        pattern = r'(.*?)\((.*?)\)'
        for detail in drug_detail:
            lis = re.findall(pattern,detail)
            if(len(lis)!=0):
                d,p = lis[0]
                all_entity["药品商"].append(d.strip(p))
                all_entity["药品"].append(p)
    all_entity = {k: list(set(v)) for k, v in all_entity.items() if k != "疾病"}
    for k in all_entity:
        if k!="疾病":
            import_entity(client,k,all_entity[k])