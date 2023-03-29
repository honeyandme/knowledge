import os
import random
with open('ner_data2.txt','r',encoding='utf-8') as f:
    data1 = f.read().split('\n')

with open('ner_dev.txt','r',encoding='utf-8') as f:
    data2 = f.read().split('\n')

with open('ner_train.txt','r',encoding='utf-8') as f:
    data3 = f.read().split('\n')

data = data1 +[""] +data2 +[""]+data3
# random.shuffle(data)
with open('all_ner_data.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(data))