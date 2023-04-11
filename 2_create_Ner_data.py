import json
import os
all_entities = {
        "疾病": [],
        "药品": [],
        "检查项目": [],
        "疾病症状": [],
    }
#将data/CMeEE-V2  (ner数据集)，转换一下数据格式。转换后的文件保存在data下的ner_train.txt和ner_dev.txt中
def load_data(path):
    with open(path,'r',encoding='utf8') as f:
        data = json.load(f)
    need_entities = ['dis','sym','dru','ite']
    mp = {'dis':'疾病','sym':'疾病症状','dru':'药品','ite':'检查项目'}
    all_text = []
    all_label = []
    for d in data:
        flag = False
        text,entities = d['text'],d['entities']
        if '发热' in text:
            print("")
        label = ['O']*len(text)
        for entity in entities:
            if(entity['type'] not in need_entities):
                continue
            all_entities[mp[entity['type']]].append(entity['entity'])
            flag = True
            label[entity['start_idx']:entity['start_idx']+1] = ['B-'+mp[entity['type']]]+['I-'+mp[entity['type']]]*(entity['end_idx']-entity['start_idx'])
        if flag:
            all_text.append(text)
            all_label.append(label)
    return all_text,all_label
def build_file(datas,labels,path):
    with open(path,'w',encoding='utf-8') as f:
        for text,label in zip(datas,labels):
            for t,l in zip(text,label):
                f.write(f'{t} {l}\n')
            f.write('\n')
if __name__== "__main__":
    train_data,train_label = load_data(os.path.join('data', 'CMeEE-V2','CMeEE-V2_train.json'))
    dev_data, dev_label = load_data(os.path.join('data', 'CMeEE-V2', 'CMeEE-V2_dev.json'))


    # build_file(train_data,train_label,os.path.join('data','ner_train.txt'))
    # build_file(train_data, train_label, os.path.join('data', 'ner_dev.txt'))
    #
    # print()
    # for name,entity in all_entities.items():
    #     with open(os.path.join('data','ent2',f'{name}.txt'),'w',encoding='utf-8') as f:
    #         entity = list(set(entity))
    #         f.write('\n'.join(entity))
