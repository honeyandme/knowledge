1. Mainfest:
- CMeEE-V2_train.json: 训练集 
- CMeEE-V2_dev.json: 验证集
- CMeEE-V2_test.json: 测试集, 选手提交的时候需要为每条记录填充"entities"字段，类型为列表。每个识别出来的实体必须包含"start_idx", "end_idx", "type"3个字段。
- example_pred.json: 提交结果示例
- README.txt: 说明文件

2. 评估指标以严格Micro-F1值为准

3. 该任务提交的文件名为：CMeEE-V2_test.json
