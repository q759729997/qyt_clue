import sys
import codecs
import random

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
    """使用同义词增强的数据扩充训练与开发数据"""
    random.seed(2020)
    train_file = './data/weibo_NER/train.conll'
    dev_file = './data/weibo_NER/dev.conll'
    # 加载数据
    data_loader = PeopleDailyNERLoader()
    data_bundle = data_loader.load({'train': train_file, 'dev': dev_file})
    print_data_bundle(data_bundle)
    # 加载扩充的数据
    train_file = './data/weibo_NER/train_augmentation.conll'
    dev_file = './data/weibo_NER/dev_augmentation.conll'
    # 加载数据
    data_loader_augmentation = PeopleDailyNERLoader()
    data_bundle_augmentation = data_loader_augmentation.load({'train': train_file, 'dev': dev_file})
    print_data_bundle(data_bundle_augmentation)
    # 组装数据
    train_data = list()
    train_data.extend([[datarow[Const.RAW_CHAR], datarow[Const.TARGET]] for datarow in data_bundle.datasets['train']])
    augmentation_list = [[datarow[Const.RAW_CHAR], datarow[Const.TARGET]] for datarow in data_bundle_augmentation.datasets['train']]
    train_data.extend(random.sample(augmentation_list, k=int(len(data_bundle.datasets['train'])/2)))
    random.shuffle(train_data)
    dev_data = list()
    dev_data.extend([[datarow[Const.RAW_CHAR], datarow[Const.TARGET]] for datarow in data_bundle.datasets['dev']])
    augmentation_list = [[datarow[Const.RAW_CHAR], datarow[Const.TARGET]] for datarow in data_bundle_augmentation.datasets['dev']]
    dev_data.extend(random.sample(augmentation_list, k=int(len(data_bundle.datasets['dev'])/2)))
    random.shuffle(dev_data)
    # 数据集切分
    logger.info('train:{}, val:{}'.format(len(train_data), len(dev_data)))
    # 数据输出
    train_augmentated_file = './data/weibo_NER/train_augmentated.conll'
    dev_augmentated_file = './data/weibo_NER/dev_augmentated.conll'
    with codecs.open(train_augmentated_file, mode='w', encoding='utf8') as fw:
        for row_id, (row_chars, target) in enumerate(train_data):
            # 输出
            for char, label in zip(row_chars, target):
                fw.write('{}\t{}\n'.format(char, label))
            fw.write('\n')
        # fw.write('\n')
    logger.info('train_augmentated_file{}'.format(train_augmentated_file))
    with codecs.open(dev_augmentated_file, mode='w', encoding='utf8') as fw:
        for row_id, (row_chars, target) in enumerate(dev_data):
            # 输出
            for char, label in zip(row_chars, target):
                fw.write('{}\t{}\n'.format(char, label))
            fw.write('\n')
        # fw.write('\n')
    logger.info('dev_augmentated_file{}'.format(dev_augmentated_file))
