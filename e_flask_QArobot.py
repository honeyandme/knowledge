import random
import d_Ner as zwk
from transformers import BertModel,BertTokenizer
import torch
import pickle
import py2neo
from flask import Flask,render_template,request
import jieba

app=Flask(__name__)
@app.route("/",methods=["GET","POST"])
def qa():
    global r
    if request.method == "GET":
        return render_template("index.html")
    else:
        inputtext = request.form.get("inputtext")
        if 'clear' in inputtext:
            r = ""
        else:
            r+="user:"+inputtext+'\n'
            entities_result = zwk.get_ner_result(model, tokenizer, inputtext, rule, tfidf_r, device, idx2tag)
            r += "robot:"+QA.ask(inputtext,entities_result)+'\n\n'
        return render_template("index.html", data=r)
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
        self.ask_cure_list = ['怎么办','怎么治','治疗方法']
        self.ask_dis_sympotm_list = ['症状','表现','怎么样']
        self.ask_no_food_list = ['忌吃', '不能吃']
        self.ask_food_list = ['吃','食']
        self.ask_check_list = ['检查']
        self.ask_acompany_list = ['并发征','并发症','导致什么','导致哪','引发什么','引发哪']
        self.ask_dis_desc_list = ['简介','来历','历史','起源','什么是']
        self.ask_dis_cause_list = ['病因','什么导致','怎么得','怎么导致','什么得','什么会得','引发']
        self.ask_dis_lasttime_list=['周期','时间','多久']
        self.ask_dis_prob_list = ['机会','率']
        self.ask_easy_get_list = ['容易得','易感']
        self.ask_dis_prevent_list = ['预防']
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
    def ask_food(self,name):
        query = "match (a:疾病{名称:'%s'})-[r:疾病宜吃食物]->(b:食物) return b.名称" % (name)
        res = self.client.run(query).data()
        res = [d['b.名称'] for d in res]
        if len(res) > 0:
            out = f'如果得了{name}，可以吃以下食物:' + ";".join(res)
        else:
            out = "抱歉！无法获取您想知道的信息。"
        if out[-1] != '。':
            out = out + '。'
        return out
    def ask_no_food(self,name):
        query = "match (a:疾病{名称:'%s'})-[r:疾病忌吃食物]->(b:食物) return b.名称" % (name)
        res = self.client.run(query).data()
        res = [d['b.名称'] for d in res]
        if len(res) > 0:
            out = f'如果得了{name}，一定不要吃以下食物:' + ";".join(res)
        else:
            out = "抱歉！无法获取您想知道的信息。"
        if out[-1] != '。':
            out = out + '。'
        return out
    def ask_check(self,name):
        query = "match (a:疾病{名称:'%s'})-[r:疾病所需检查]->(b:检查项目) return b.名称" % (name)
        res = self.client.run(query).data()
        res = [d['b.名称'] for d in res]
        if len(res) > 0:
            out = f'如果得了{name}，可以做如下检查:' + ";".join(res)
        else:
            out = "抱歉！无法获取您想知道的信息。"
        if out[-1] != '。':
            out = out + '。'
        return out
    def ask_acompany(self,name):
        query = "match (a:疾病{名称:'%s'})-[r:疾病并发疾病]->(b:疾病) return b.名称" % (name)
        res = self.client.run(query).data()
        res = [d['b.名称'] for d in res]
        if len(res) > 0:
            out = f'{name}的并发疾病主要有:' + ";".join(res)
        else:
            out = "抱歉！无法获取您想知道的信息。"
        if out[-1] != '。':
            out = out + '。'
        return out
    def ask_dis_desc(self,name):

        query = "match (a:疾病{名称:'%s'}) return a.疾病简介" % (name)
        res = self.client.run(query).data()[0].values()
        if len(res) > 0:
            out = "。".join(res)
        else:
            out = "抱歉！无法获取您想知道的信息。"
        if out[-1] != '。':
            out = out + '。'
        return out
    def ask_dis_cause(self,name):
        query = "match (a:疾病{名称:'%s'}) return a.疾病病因" % (name)
        res = self.client.run(query).data()[0].values()
        if len(res) > 0:
            out = "。".join(res)
        else:
            out = "抱歉！无法获取您想知道的信息。"
        if out[-1] != '。':
            out = out + '。'
        return out
    def ask_dis_lasttime(self,name):
        query = "match (a:疾病{名称:'%s'}) return a.治疗周期" % (name)
        res = self.client.run(query).data()[0].values()
        if len(res) > 0:
            out = "。".join(res)
        else:
            out = "抱歉！无法获取您想知道的信息。"
        if out[-1] != '。':
            out = out + '。'
        return out
    def ask_dis_prob(self,name):
        query = "match (a:疾病{名称:'%s'}) return a.治愈概率" % (name)
        res = self.client.run(query).data()[0].values()
        if len(res) > 0:
            out = "。".join(res)
        else:
            out = "抱歉！无法获取您想知道的信息。"
        if out[-1] != '。':
            out = out + '。'
        return out
    def ask_easy_get(self,name):
        query = "match (a:疾病{名称:'%s'}) return a.疾病易感人群" % (name)
        res = self.client.run(query).data()[0].values()
        if len(res) > 0:
            out = "。".join(res)
        else:
            out = "抱歉！无法获取您想知道的信息。"
        if out[-1] != '。':
            out = out + '。'
        return out
    def ask_dis_prevent(self,name):
        query = "match (a:疾病{名称:'%s'}) return a.预防措施" % (name)
        res = self.client.run(query).data()[0].values()
        if len(res) > 0:
            out = "。".join(res)
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

        is_ask_no_food = self.keyword_in(sen,self.ask_no_food_list)
        if is_ask_no_food:
            if "疾病" in en_list:
                return random.choice(self.hello)+self.ask_no_food(en_list.get("疾病",""))
            elif "疾病症状" in en_list:
                dis_name = self.ask_dis_name(en_list.get("疾病症状",""))
                if len(dis_name)==0:
                    return random.choice(self.hello) + '我不知道您为什么会'+en_list.get("疾病症状","")
                out = random.choice(self.hello)+'我推测您可能是得了{%s}'%(dis_name) +'。'
                out += self.ask_no_food(dis_name)
                return out

        is_ask_food = self.keyword_in(sen,self.ask_food_list)
        if is_ask_food:
            if "疾病" in en_list:
                return random.choice(self.hello)+self.ask_food(en_list.get("疾病",""))
            elif "疾病症状" in en_list:
                dis_name = self.ask_dis_name(en_list.get("疾病症状",""))
                if len(dis_name)==0:
                    return random.choice(self.hello) + '我不知道您为什么会'+en_list.get("疾病症状","")
                out = random.choice(self.hello)+'我推测您可能是得了{%s}'%(dis_name) +'。'
                out += self.ask_food(dis_name)
                return out

        is_ask_check = self.keyword_in(sen,self.ask_check_list)
        if is_ask_check:
            if "疾病" in en_list:
                return random.choice(self.hello)+self.ask_check(en_list.get("疾病",""))
            elif "疾病症状" in en_list:
                dis_name = self.ask_dis_name(en_list.get("疾病症状",""))
                if len(dis_name)==0:
                    return random.choice(self.hello) + '我不知道您为什么会'+en_list.get("疾病症状","")
                out = random.choice(self.hello)+'我推测您可能是得了{%s}'%(dis_name) +'。'
                out += self.ask_check(dis_name)
                return out

        is_ask_acompany = self.keyword_in(sen,self.ask_acompany_list)
        if is_ask_acompany:
            if "疾病" in en_list:
                return random.choice(self.hello)+self.ask_acompany(en_list.get("疾病",""))
            elif "疾病症状" in en_list:
                dis_name = self.ask_dis_name(en_list.get("疾病症状",""))
                if len(dis_name)==0:
                    return random.choice(self.hello) + '我不知道您为什么会'+en_list.get("疾病症状","")
                out = random.choice(self.hello)+'我推测您可能是得了{%s}'%(dis_name) +'。'
                out += self.ask_acompany(dis_name)
                return out


        is_ask_dis_desc = self.keyword_in(sen,self.ask_dis_desc_list)
        if is_ask_dis_desc:
            if "疾病" in en_list:
                return random.choice(self.hello)+self.ask_dis_desc(en_list.get("疾病",""))
            elif "疾病症状" in en_list:
                dis_name = self.ask_dis_name(en_list.get("疾病症状",""))
                if len(dis_name)==0:
                    return random.choice(self.hello) + '我不知道您为什么会'+en_list.get("疾病症状","")
                out = random.choice(self.hello)+'我推测您可能是得了{%s}'%(dis_name) +'。'
                out += self.ask_dis_desc(dis_name)
                return out

        is_ask_dis_cause = self.keyword_in(sen,self.ask_dis_cause_list)
        if is_ask_dis_cause:
            if "疾病" in en_list:
                return random.choice(self.hello)+self.ask_dis_cause(en_list.get("疾病",""))
            elif "疾病症状" in en_list:
                dis_name = self.ask_dis_name(en_list.get("疾病症状",""))
                if len(dis_name)==0:
                    return random.choice(self.hello) + '我不知道您为什么会'+en_list.get("疾病症状","")
                out = random.choice(self.hello)+'我推测您可能是得了{%s}'%(dis_name) +'。'
                out += self.ask_dis_cause(dis_name)
                return out

        is_ask_dis_lasttime = self.keyword_in(sen,self.ask_dis_lasttime_list)
        if is_ask_dis_lasttime:
            if "疾病" in en_list:
                return self.ask_dis_lasttime(en_list.get("疾病",""))
            elif "疾病症状" in en_list:
                dis_name = self.ask_dis_name(en_list.get("疾病症状",""))
                if len(dis_name)==0:
                    return random.choice(self.hello) + '我不知道您为什么会'+en_list.get("疾病症状","")
                out = random.choice(self.hello)+'我推测您可能是得了{%s}'%(dis_name) +'。'
                out += self.ask_dis_lasttime(dis_name)
                return out

        is_ask_dis_prob = self.keyword_in(sen,self.ask_dis_prob_list)
        if is_ask_dis_prob:
            if "疾病" in en_list:
                return self.ask_dis_prob(en_list.get("疾病",""))
            elif "疾病症状" in en_list:
                dis_name = self.ask_dis_name(en_list.get("疾病症状",""))
                if len(dis_name)==0:
                    return random.choice(self.hello) + '我不知道您为什么会'+en_list.get("疾病症状","")
                out = random.choice(self.hello)+'我推测您可能是得了{%s}'%(dis_name) +'。'
                out += self.ask_dis_prob(dis_name)
                return out


        is_ask_easy_get = self.keyword_in(sen,self.ask_easy_get_list)
        if is_ask_easy_get:
            if "疾病" in en_list:
                return self.ask_easy_get(en_list.get("疾病",""))
            elif "疾病症状" in en_list:
                dis_name = self.ask_dis_name(en_list.get("疾病症状",""))
                if len(dis_name)==0:
                    return random.choice(self.hello) + '我不知道您为什么会'+en_list.get("疾病症状","")
                out = random.choice(self.hello)+'我推测您可能是得了{%s}'%(dis_name) +'。'
                out += self.ask_easy_get(dis_name)
                return out

        is_ask_dis_prevent = self.keyword_in(sen,self.ask_dis_prevent_list)
        if is_ask_dis_prevent:
            if "疾病" in en_list:
                return self.ask_dis_prevent(en_list.get("疾病",""))
            elif "疾病症状" in en_list:
                dis_name = self.ask_dis_name(en_list.get("疾病症状",""))
                if len(dis_name)==0:
                    return random.choice(self.hello) + '我不知道您为什么会'+en_list.get("疾病症状","")
                out = random.choice(self.hello)+'我推测您可能是得了{%s}'%(dis_name) +'。'
                out += self.ask_dis_prevent(dis_name)
                return out
        return "抱歉，我无法识别您的问题～"






if __name__=="__main__":
    client = py2neo.Graph('http://localhost:7474', user='neo4j', password='wei8kang7.long', name='neo4j')

    rule, tfidf_r, tokenizer, model,device,tag2idx = get_pre_tool()
    idx2tag = list(tag2idx)
    QA = QuestionCls(client)
    r = ""
    app.run(host="127.0.0.1", port=9999, debug=True)

    # while(True):
    #     sen = input('请输入:')
    #     entities_result = zwk.get_ner_result(model, tokenizer, sen, rule, tfidf_r,device,idx2tag)
    #     print(entities_result)
    #     res = QA.ask(sen,entities_result)
    #     print(res)
