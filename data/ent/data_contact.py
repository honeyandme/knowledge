import os

dir = os.listdir(os.path.join('..','ent1'))

for en in dir:
    with open(os.path.join('..','ent1',en),'r',encoding='utf-8') as f:
        text = f.read().split('\n')
    if os.path.exists(os.path.join('..','ent2',en)):
        with open(os.path.join('..', 'ent2', en), 'r', encoding='utf-8') as f:
            text += f.read().split('\n')
    text = list(set(text))#去重
    text = [te for te in text if len(te)>=1]
    with open(os.path.join('..', 'ent', en), 'w', encoding='utf-8') as f:
        f.write('\n'.join(text))
