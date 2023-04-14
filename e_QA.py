import random
import d_Ner as zwk
from transformers import BertModel,BertTokenizer
import torch
import pickle
import py2neo
import random
def get_pre_tool():
    rule = zwk.rule_find()
    tfidf_r = zwk.tfidf_alignment()

    model_name = 'hfl/chinese-roberta-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = zwk.Bert_Model(model_name, hidden_size=128, tag_num=16, bi=True)
    model.load_state_dict(torch.load('best_model.pt'))
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model = model.to(device)

    with open('tag2idx.npy', 'rb') as f:
        tag2idx = pickle.load(f)

    return rule,tfidf_r,tokenizer,model,device,tag2idx

class QuestionCls():
    def __init__(self,client):
        self.client = client
        self.ask_dru_list = ['什么药','哪种药','啥药']
        self.ask_cure_list = ['怎么办','治疗']
        self.ask_dis_sympotm_list = ['症状','表现','怎么样']
        self.ask_food_list = []
        self.hello = ['您好！','你好！','你好,我是聊天机器人！','您好呀！']
    def ask_cure_method(self,name):
        query1 = "match (a:疾病{名称:'%s'})-[r:治疗的方法]->(b:治疗方法) return b.名称" % (name)
        # query2 = "match (a:疾病{名称:'%s'})-[r:治疗的方法]->(b:治疗方法) return b.名称" % (en_list['疾病症状'])
        res = self.client.run(query1).data()
        res = [d['b.名称'] for d in res]
        if len(res) > 0:
            out = f'如果您患有{name}，可以采用的治疗手段有:' + ";".join(res)
        else:
            out = "抱歉！无法获取您想知道的信息。"
        if out[-1] != '。':
            out = out + '。'
        return out
    def ask_dis_name(self,name):
        query = "match (a:疾病)-[r:疾病的症状]->(b:疾病症状{名称:'%s'}) return a.名称" % (name)
        res = self.client.run(query).data()
        if len(res)==0:
            return ""
        res = res[0]['a.名称']
        return res
    def ask_dis_symptom(self,name):
        query = "match (a:疾病{名称:'%s'})-[r:疾病的症状]->(b:疾病症状) return b.名称" % (name)
        res = self.client.run(query).data()
        res = [d['b.名称'] for d in res]
        if len(res) > 0:
            out = f'如果得了{name}，可能会有如下症状:' + ";".join(res)
        else:
            out = "抱歉！无法获取您想知道的信息。"
        if out[-1] != '。':
            out = out + '。'
        return out
    def ask_dru(self,name):
        query = "match (a:疾病{名称:'%s'})-[r:疾病使用药品]->(b:药品) return b.名称" % (name)
        res = self.client.run(query).data()
        res = [d['b.名称'] for d in res]
        if len(res) > 0:
            out = f'如果得了{name}，可以尝试这些药品:' + ";".join(res)
        else:
            out = "抱歉！无法获取您想知道的信息。"
        if out[-1] != '。':
            out = out + '。'
        return out
    def keyword_in(self,sen,keyword):
        for word in keyword:
            if word in sen:
                return True
        return False
    def ask(self,sen,en_list):
        is_ask_dru_list = self.keyword_in(sen,self.ask_cure_list)
        if is_ask_dru_list:
            if "疾病" in en_list:
                return random.choice(self.hello) + self.ask_cure_method(en_list.get("疾病",""))
            elif "疾病症状" in en_list:
                dis_name = self.ask_dis_name(en_list.get("疾病症状",""))
                if len(dis_name)==0:
                    return random.choice(self.hello) + '我不知道您为什么会'+en_list.get("疾病症状","")
                out = random.choice(self.hello)+'我推测您可能是得了{%s}'%(dis_name) +'。'
                out += self.ask_cure_method(dis_name)
                return out

        is_ask_dis_symptom = self.keyword_in(sen,self.ask_dis_sympotm_list)
        if is_ask_dis_symptom:
            if "疾病" in en_list:
                return self.ask_dis_symptom(en_list["疾病"])

        is_ask_dru = self.keyword_in(sen,self.ask_dru_list)
        if is_ask_dru:
            if "疾病" in en_list:
                return random.choice(self.hello)+self.ask_dru(en_list.get("疾病",""))
            elif "疾病症状" in en_list:
                dis_name = self.ask_dis_name(en_list.get("疾病症状",""))
                if len(dis_name)==0:
                    return random.choice(self.hello) + '我不知道您为什么会'+en_list.get("疾病症状","")
                out = random.choice(self.hello)+'我推测您可能是得了{%s}'%(dis_name) +'。'
                out += self.ask_dru(dis_name)
                return out
        return "抱歉，我无法识别您的问题～"






if __name__=="__main__":
    client = py2neo.Graph('http://localhost:7474', user='neo4j', password='wei8kang7.long', name='neo4j')

    rule, tfidf_r, tokenizer, model,device,tag2idx = get_pre_tool()
    idx2tag = list(tag2idx)
    QA = QuestionCls(client)
    while(True):
        sen = input('请输入:')
        entities_result = zwk.get_ner_result(model, tokenizer, sen, rule, tfidf_r,device,idx2tag)
        print(entities_result)
        res = QA.ask(sen,entities_result)
        print(res)
