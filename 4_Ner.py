import torch
from torch import nn
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from transformers import BertModel,BertTokenizer
def get_data(path):
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
    return all_text,all_tag
class Nerdataset(Dataset):
    def __init__(self,all_text,all_label,tokenizer):
        self.all_text = all_text
        self.all_label = all_label
        self.tokenizer = tokenizer
    def __getitem__(self, x):
        pass
    def __len__(self):
        return len(self.all_text)
if __name__ == "__main__":
    all_text,all_label = get_data(os.path.join('data','prodata','all_ner_data.txt'))
    train_text, dev_text, train_label, dev_label = train_test_split(all_text, all_text, test_size = 0.02, random_state = 42)

    max_len = 50
    epoch = 1
    batch_size = 40
    hidden_num = 128
    model_name='./bert_base_chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = Nerdataset(train_text,train_label,tokenizer)

    # for e in range(epoch):
    print()