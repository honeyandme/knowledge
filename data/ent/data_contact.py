import os
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
dir = os.listdir(os.path.join('..','ent1'))
for en in dir:
    path1 = os.path.join('..','ent1',en)
    en_name = en.strip('.txt')
    if os.path.exists(path1):
        with open(path1,'r',encoding='utf-8') as f:
            data = f.read().split('\n')
            for d in data:
                d = d.split(' ')
                if(len(d)!=2):
                    continue
                if d[0] in cipin[en_name]:
                    cipin[en_name][d[0]] += int(d[1])
                else:
                    try:
                        cipin[en_name][d[0]] = int(d[1])
                    except:
                        print("")
for en in dir:
    path1 = os.path.join('..', 'ent2', en)
    en_name = en.strip('.txt')
    if os.path.exists(path1):
        with open(path1,'r',encoding='utf-8') as f:
            data = f.read().split('\n')
            for d in data:
                d = d.split(' ')
                if len(d)!=2:
                    continue
                if d[0] in cipin[en_name]:
                    cipin[en_name][d[0]] += int(d[1])*10
                else:
                    cipin[en_name][d[0]] = int(d[1])*10
for name, entity in cipin.items():
    with open(os.path.join( f'{name}.txt'), 'w', encoding='utf-8') as f:
        entity = sorted(entity.items(), key=lambda x: int(x[1]), reverse=True)
        for en, num in entity:
            num = int(num)
            if num>=1000:
                num = num//30
            elif num>=100:
                num = num//10
            elif num>=10:
                num = num//2

            f.write(f"{en} {num}\n")
# dir = os.listdir(os.path.join('..','ent1'))
#
# for en in dir:
#     with open(os.path.join('..','ent1',en),'r',encoding='utf-8') as f:
#         text = f.read().split('\n')
#     if os.path.exists(os.path.join('..','ent2',en)):
#         with open(os.path.join('..', 'ent2', en), 'r', encoding='utf-8') as f:
#             text += f.read().split('\n')
#     text = list(set(text))#去重
#     text = [te for te in text if len(te)>=1]
#     with open(os.path.join('..', 'ent', en), 'w', encoding='utf-8') as f:
#         f.write('\n'.join(text))
