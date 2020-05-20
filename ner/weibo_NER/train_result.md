# 模型训练结果

- `random`采用100维随机初始化的词嵌入；
- `word2vec`采用`cn_char_fastnlp_100d`；

| 词嵌入           | 模型       | 测试集表现（span F值；3次测试） | 平均F值 |
| ---------------- | ---------- | ------------------------------- | ------- |
| random           | bilstm-crf | 0.386399；0.41954；0.394118     |         |
| word2vec         | bilstm-crf | 0.440233；0.405844；0.436533    |         |
| bert(不更新参数) | bilstm-crf | 0.686224；0.640523；0.647292    |         |
| bert(更新参数)   | bert-crf   | 0.691076；0.669031；0.681657    |         |

- 训练数据使用最优模型预测后的数据

| 词嵌入           | 模型       | 测试集表现（span F值；3次测试） | 平均F值 |
| ---------------- | ---------- | ------------------------------- | ------- |
| random           | bilstm-crf |                                 |         |
| word2vec         | bilstm-crf | 0.459701；0.458824；            |         |
| bert(不更新参数) | bilstm-crf |                                 |         |