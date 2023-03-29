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
class Nerdataset(Dataset):
    def __init__(self,all_text,all_label,tokenizer,max_len,tag2idx,is_dev=False):
        self.all_text = all_text
        self.all_label = all_label
        self.tokenizer = tokenizer
        self.max_len= max_len
        self.tag2idx = tag2idx
        self.is_dev = is_dev
    def __getitem__(self, x):
        if self.is_dev:
            max_len = len(self.all_text[x])+2
        else:
            max_len = self.max_len
        text,label = self.all_text[x][:max_len-2],self.all_label[x][:max_len-2]
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
        for x,y,batch_len in tqdm(dev_dataloader):
            x = x.to(device)
            pre = model(x)
            pre = [idx2tag[i] for i in pre[1:batch_len+1]]
            all_pre.append(pre)

        print(f'e={e},loss={loss_sum/ba:.5f} f1={f1_score(all_pre,dev_label)}')


