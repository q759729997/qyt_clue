import sys
import codecs

from sklearn.model_selection import train_test_split
from fastNLP.core import Const
from fastNLP.io import PeopleDailyNERLoader

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.core.utils import print_data_bundle  # noqa
from myClue.tools.serialize import load_serialize_obj  # noqa
from myClue.tools.decoder import decode_ner_tags  # noqa


if __name__ == "__main__":
    """训练和dev重新划分"""
    train_file = './data/weibo_NER/train.conll'
    dev_file = './data/weibo_NER/dev.conll'
    # 加载数据
    data_loader = PeopleDailyNERLoader()
    data_bundle = data_loader.load({'train': train_file, 'dev': dev_file})
    print_data_bundle(data_bundle)
    # 组装数据
    data_list = list()
    data_list.extend([[datarow[Const.RAW_CHAR], datarow[Const.TARGET]] for datarow in data_bundle.datasets['train']])
    data_list.extend([[datarow[Const.RAW_CHAR], datarow[Const.TARGET]] for datarow in data_bundle.datasets['dev']])
    # 数据集切分
    train_data, dev_data = train_test_split(data_list, test_size=len(data_bundle.datasets['dev']), shuffle=True, random_state=2020)
    logger.info('数据切分结果： all:{}, train:{}, val:{}'.format(len(data_list), len(train_data), len(dev_data)))
    # 数据输出
    train_resplit_file = './data/weibo_NER/train_resplit.conll'
    dev_resplit_file = './data/weibo_NER/dev_resplit.conll'
    with codecs.open(train_resplit_file, mode='w', encoding='utf8') as fw:
        for row_id, (row_chars, target) in enumerate(train_data):
            # 输出
            for char, label in zip(row_chars, target):
                fw.write('{}\t{}\n'.format(char, label))
            fw.write('\n')
        # fw.write('\n')
    logger.info('train_resplit_file{}'.format(train_resplit_file))
    with codecs.open(dev_resplit_file, mode='w', encoding='utf8') as fw:
        for row_id, (row_chars, target) in enumerate(dev_data):
            # 输出
            for char, label in zip(row_chars, target):
                fw.write('{}\t{}\n'.format(char, label))
            fw.write('\n')
        # fw.write('\n')
    logger.info('dev_resplit_file{}'.format(dev_resplit_file))
