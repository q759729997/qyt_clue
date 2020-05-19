# 短文本分类

- 数据集与任务网站： <https://www.cluebenchmarks.com/introduce.html>

## 数据集介绍

- 数据集名称：Short Text Classification for News
- 每一条数据有三个属性，从前往后分别是 分类ID，分类名称，新闻字符串（仅含标题）。
- 数据量：训练集(266,000)，验证集(57,000)，测试集(57,000),例子：

~~~shell
{"label": "102", "label_des": "news_entertainment", "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物"}
~~~

## issue

- 尝试了一个textcnn模型。该数据噪音较多，无法准备评估模型效果。