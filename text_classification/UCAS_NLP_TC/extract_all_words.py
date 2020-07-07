"""提取train中的所有单词"""
import codecs
import sys
from tqdm import tqdm
from collections import OrderedDict
sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.tools.file import read_file_texts  # noqa
from myClue.tools.file import init_file_path  # noqa


if __name__ == "__main__":
    # train_file = './data/UCAS_NLP_TC/traindata.txt'
    # output_file = './data/UCAS_NLP_TC/train_words.txt'
    train_file = './data/UCAS_NLP_TC/data_11_baidu_basicwords/traindata.txt'
    output_file = './data/UCAS_NLP_TC/train_baidu_basicwords.txt'
    texts = read_file_texts(train_file)
    with codecs.open(output_file, mode='w', encoding='utf8') as fw:
        words = OrderedDict()
        for text in tqdm(texts):
            label, news_content = text.split('\t')
            row_words = news_content.split(' ')
            for word in row_words:
                words[word] = None
        for word in tqdm(words):
            fw.write('{}\n'.format(word))
