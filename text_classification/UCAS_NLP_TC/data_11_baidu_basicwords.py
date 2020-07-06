# coding:utf-8
"""训练数据与测试数据随机排序,百度细粒度分词结果
"""
import codecs
import os
import sys
import json
from tqdm import tqdm

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.tools.file import read_file_texts  # noqa
from myClue.tools.file import init_file_path  # noqa


if __name__ == "__main__":
    train_file_config = {
        'train': './data/UCAS_NLP_TC/data_baidu_cws/train_cws.json',
        'dev': './data/UCAS_NLP_TC/data_baidu_cws/dev_cws.json',
        'test': './data/UCAS_NLP_TC/data_baidu_cws/test_cws.json',
    }
    output_path = './data/UCAS_NLP_TC/data_11_baidu_basicwords'
    init_file_path(output_path)
    for file_label, file_name in train_file_config.items():
        logger.info('开始处理：{}'.format(file_label))
        texts = read_file_texts(file_name)
        output_file_name = os.path.join(output_path, '{}data.txt'.format(file_label))
        with codecs.open(output_file_name, mode='w', encoding='utf8') as fw:
            for text in tqdm(texts):
                row_data = json.loads(text)
                label = row_data['label']
                cws_items = row_data['cws_items']
                words = list()
                for cws_item in cws_items:
                    words.extend(cws_item['basic_words'])
                news_content = ' '.join(words)
                if len(news_content) == 0:
                    continue
                fw.write('{}\t{}\n'.format(label, news_content))
