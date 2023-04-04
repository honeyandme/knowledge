import random

import torch
from torch import nn
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from transformers import BertModel,BertTokenizer
from tqdm import tqdm
from seqeval.metrics import f1_score
def get_data(path,max_len=None):
    all_text,all_tag = [],[]
    with open(path,'r',encoding='utf8') as f:
        all_data = f.read().split('\n')

    sen,tag = [],[]
    for data in all_data:
        data = data.split(' ')
        if(len(data)!=2):
            if len(sen)>2:
                all_text.append(sen)
                all_tag.append(tag)
            sen, tag = [], []
            continue
        te,ta = data
        sen.append(te)
        tag.append(ta)
    if max_len is not None:
        return all_text[:max_len], all_tag[:max_len]
    return all_text,all_tag



#找出tag(label)中的所有实体及其下表，为实体动态替换/随机掩码策略/实体动态拼接做准备
def find_entities(tag):
    result = []#[(2,3,'药品'),(7,10,'药品商')]
    label_len = len(tag)
    i = 0
    while(i<label_len):
        if(tag[i][0]=='B'):
            type = tag[i].strip('B-')
            j=i+1
            while(j<label_len and tag[j][0]=='I'):
                j += 1
            result.append((i,j-1,type))
            i=j
        else:
            i = i + 1
    return result




class Entity_Extend:
    def __init__(self):
        eneities_path = os.path.join('data','ent')
        files = os.listdir(eneities_path)
        files = [docu for docu in files if '.py' not in docu]

        self.type2entity = {}

        for type in files:
            with open(os.path.join(eneities_path,type),'r',encoding='utf-8') as f:
                entities = f.read().split('\n')
                type = type.strip('.txt')
                self.type2entity[type] = entities
    def no_work(self,te,tag,type):
        return te,tag

    # 1. 实体替换
    def entity_replace(self,te,ta,type):
        choice_ent = random.choice(self.type2entity[type])
        ta = ["B-"+type] + ["I-"+type]*(len(choice_ent)-1)
        return list(choice_ent),ta

    # 2. 实体掩盖
    def entity_mask(self,te,ta,type):
        if(len(te)<=3):
            return te,ta
        elif(len(te)<=5):
            te.pop(random.randint(0,len(te)-1))
        else:
            te.pop(random.randint(0, len(te) - 1))
            te.pop(random.randint(0, len(te) - 1))
        ta = ["B-" + type] + ["I-" + type] * (len(te) - 1)
        return te,ta

    # 3. 实体拼接
    def entity_union(self,te,ta,type):
        words = ['和','与','以及','还有']
        wor = random.choice(words)
        choice_ent = random.choice(self.type2entity[type])
        te = te+list(wor)+list(choice_ent)
        ta = ta+['O']*len(wor)+["B-"+type] + ["I-"+type]*(len(choice_ent)-1)
        return te,ta
    def entities_extend(self,text,tag,ents):
        cho = [self.no_work,self.no_work,self.entity_replace,self.entity_mask,self.entity_union,self.entity_union]
        new_text = text.copy()
        new_tag = tag.copy()
        sign = 0
        for ent in ents:
            p = random.choice(cho)
            te,ta = p(text[ent[0]:ent[1]+1],tag[ent[0]:ent[1]+1],ent[2])
            new_text[ent[0] + sign:ent[1] + 1 + sign], new_tag[ent[0] + sign:ent[1] + 1 + sign] = te,ta
            sign += len(te)-(ent[1]-ent[0]+1)

        return new_text, new_tag




class Nerdataset(Dataset):
    def __init__(self,all_text,all_label,tokenizer,max_len,tag2idx,is_dev=False):
        self.all_text = all_text
        self.all_label = all_label
        self.tokenizer = tokenizer
        self.max_len= max_len
        self.tag2idx = tag2idx
        self.is_dev = is_dev
        self.entity_extend = Entity_Extend()
    def __getitem__(self, x):
        text, label = self.all_text[x], self.all_label[x]
        if self.is_dev:
            max_len = min(len(self.all_text[x])+2,500)
        else:
            # 几种策略
            ents = find_entities(label)
            text,label = self.entity_extend.entities_extend(text,label,ents)
            max_len = self.max_len
        text, label =text[:max_len - 2], label[:max_len - 2]

        x_len = len(text)
        assert len(text)==len(label)
        text_idx = self.tokenizer.encode(text,add_special_token=True)
        label_idx = [self.tag2idx['<PAD>']] + [self.tag2idx[i] for i in label] + [self.tag2idx['<PAD>']]

        text_idx +=[0]*(max_len-len(text_idx))
        label_idx +=[self.tag2idx['<PAD>']]*(max_len-len(label_idx))
        return torch.tensor(text_idx),torch.tensor(label_idx),x_len
    def __len__(self):
        return len(self.all_text)




def build_tag2idx(all_tag):
    tag2idx = {'<PAD>':0}
    for sen in all_tag:
        for tag in sen:
            tag2idx[tag] = tag2idx.get(tag,len(tag2idx))
    return tag2idx




class Bert_Model(nn.Module):
    def __init__(self,model_name,hidden_size,tag_num,bi):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.gru = nn.RNN(input_size=768,hidden_size=hidden_size,num_layers=1,batch_first=True,bidirectional=bi)
        if bi:
            self.classifier = nn.Linear(hidden_size*2,tag_num)
        else:
            self.classifier = nn.Linear(hidden_size, tag_num)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    def forward(self,x,label=None):
        bert_0,_ = self.bert(x,attention_mask=(x>0),return_dict=False)
        gru_0,_ = self.gru(bert_0)
        pre = self.classifier(gru_0)
        if label is not None:
            loss = self.loss_fn(pre.reshape(-1,pre.shape[-1]),label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre,dim=-1).squeeze(0)





if __name__ == "__main__":
    all_text,all_label = get_data(os.path.join('data','prodata','all_ner_data.txt'),10000)
    train_text, dev_text, train_label, dev_label = train_test_split(all_text, all_label, test_size = 0.02, random_state = 42)
    tag2idx = build_tag2idx(all_label)
    idx2tag = list(tag2idx)

    max_len = 50
    epoch = 10
    batch_size = 40
    hidden_size = 128
    bi = True
    model_name='./bert_base_chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    lr =1e-5

    device = torch.device('mps') if torch.backends.mps.is_available()   else torch.device('cpu')
    # device = torch.device('cpu')

    train_dataset = Nerdataset(train_text,train_label,tokenizer,max_len,tag2idx)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    dev_dataset = Nerdataset(dev_text, dev_label, tokenizer, max_len, tag2idx,is_dev=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False)

    model = Bert_Model(model_name,hidden_size,len(tag2idx),bi).to(device)
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    for e in range(epoch):
        loss_sum = 0
        ba = 0
        for x,y,batch_len in tqdm(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            loss = model(x,y)
            loss.backward()

            opt.step()
            loss_sum+=loss
            ba += 1
        all_pre = []
        all_label = []
        for x,y,batch_len in tqdm(dev_dataloader):
            assert len(x)==len(y)
            x = x.to(device)
            pre = model(x)
            pre = [idx2tag[i] for i in pre[1:batch_len+1]]
            all_pre.append(pre)

            label = [idx2tag[i] for i in y[0][1:batch_len+1]]
            all_label.append(label)
        print(f'e={e},loss={loss_sum/ba:.5f} f1={f1_score(all_pre,all_label)}')


