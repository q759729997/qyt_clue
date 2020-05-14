"""
将数据转换为文本文件形式；并生成模型所需格式。
此文件问可执行脚本，包含main入口函数。
"""
import codecs
import os
import sys

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.tools.file import read_json_file_iter  # noqa
from myClue.tools.text import remove_blank  # noqa


if __name__ == "__main__":
    file_path = './data/tnews_public'
    file_types = ('train', 'dev', 'test')
    for file_type in file_types:
        logger.info('开始处理:{}'.format(file_type))
        json_file_name = os.path.join(file_path, '{}.json'.format(file_type))
        logger.info('json文件名:{}'.format(json_file_name))
        json_file_iter = read_json_file_iter(json_file_name)
        txt_file_name = os.path.join(file_path, '{}.txt'.format(file_type))
        row_count = 0
        with codecs.open(txt_file_name, mode='w', encoding='utf8') as fw:
            for row_json in json_file_iter:
                label = row_json.get('label_desc', None)
                sentence = row_json.get('sentence', '')
                keywords = row_json.get('keywords', '')
                text = remove_blank('{}{}'.format(sentence, keywords))
                if label is None:
                    fw.write('{}\n'.format(text))
                else:
                    fw.write('{}\t{}\n'.format(label, text))
                row_count += 1
        logger.info('处理完毕，输出文件:{}'.format(txt_file_name))
        logger.info('数据量:{}'.format(row_count))
