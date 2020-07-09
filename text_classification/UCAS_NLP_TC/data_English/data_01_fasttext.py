# coding:utf-8
"""fasttext数据集
"""
import codecs
import os
import sys
import json
import random
from tqdm import tqdm

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.tools.file import read_file_texts  # noqa
from myClue.tools.file import init_file_path  # noqa


if __name__ == "__main__":
    train_file_config = {
        'train': './data/UCAS_NLP_TC/data_baidu_trans/train_trans.json',
        'dev': './data/UCAS_NLP_TC/data_baidu_trans/dev_trans.json',
        'test': './data/UCAS_NLP_TC/data_baidu_trans/test_trans.json',
    }
    output_path = './data/UCAS_NLP_TC/data_English/data_01_fasttext'
    init_file_path(output_path)
    texts = read_file_texts(train_file_config['train'])
    texts.extend(read_file_texts(train_file_config['dev']))
    random.shuffle(texts)
    output_file_name = os.path.join(output_path, 'train_data.txt')
    with codecs.open(output_file_name, mode='w', encoding='utf8') as fw:
        for text in tqdm(texts):
            row_data = json.loads(text)
            label = row_data['label']
            trans_results = row_data['trans_results']
            eng_texts = list()
            for trans_result in trans_results:
                eng_texts.append(trans_result['dst'])
            news_content = ' '.join(eng_texts)
            if len(news_content) == 0:
                continue
            fw.write('__label__{}\t{}\n'.format(label, news_content))
    texts = read_file_texts(train_file_config['test'])
    # random.shuffle(texts)
    output_file_name = os.path.join(output_path, 'test_data.txt')
    with codecs.open(output_file_name, mode='w', encoding='utf8') as fw:
        for text in tqdm(texts):
            row_data = json.loads(text)
            label = row_data['label']
            trans_results = row_data['trans_results']
            eng_texts = list()
            for trans_result in trans_results:
                eng_texts.append(trans_result['dst'])
            news_content = ' '.join(eng_texts)
            if len(news_content) == 0:
                continue
            fw.write('__label__{}\t{}\n'.format(label, news_content))
