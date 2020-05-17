# 我的CLUE实践

## CLUE资料

- CLUE:中文任务基准测评 Chinese Language Understanding Evaluation Benchmark: datasets, baselines, pre-trained models, corpus and leaderboard
- CLUE官网：<https://www.cluebenchmarks.com/index.html>
- github地址：<https://github.com/CLUEbenchmark/CLUE>

## ChineseGLUE资料

- Language Understanding Evaluation benchmark for Chinese: datasets, baselines, pre-trained models, corpus and leaderboard
- 中文语言理解测评基准，包括代表性的数据集、基准(预训练)模型、语料库、排行榜。
- github地址： <https://github.com/chineseGLUE/chineseGLUE>

## 环境准备

- conda环境:

~~~shell
conda create --name clue python=3.6
~~~

- pytorch依赖安装：

~~~shell
pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
~~~

- 其他依赖：

~~~shell
pip install -r requirements.txt -i https://pypi.douban.com/simple
~~~