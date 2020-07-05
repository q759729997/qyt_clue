"""训练数据与测试数据随机排序，使用字特征"""
import codecs
import os
import sys
from tqdm import tqdm
sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.tools.file import read_file_texts  # noqa
from myClue.tools.file import init_file_path  # noqa


def news_content_process(news_content):
    """"数据转换处理"""
    news_content = news_content.replace(' ', '')
    return ' '.join(list(news_content))


if __name__ == "__main__":
    train_file_config = {
        'train': './data/UCAS_NLP_TC/data_01_shuffle/traindata.txt',
        'dev': './data/UCAS_NLP_TC/data_01_shuffle/devdata.txt',
    }
    output_path = './data/UCAS_NLP_TC/data_02_chars'
    init_file_path(output_path)
    for file_label, file_name in train_file_config.items():
        logger.info('开始处理：{}'.format(file_label))
        texts = read_file_texts(file_name)
        output_file_name = os.path.join(output_path, '{}data.txt'.format(file_label))
        with codecs.open(output_file_name, mode='w', encoding='utf8') as fw:
            for text in tqdm(texts):
                label, news_content = text.split('\t')
                news_content = news_content_process(news_content)
                fw.write('{}\t{}\n'.format(label, news_content))
