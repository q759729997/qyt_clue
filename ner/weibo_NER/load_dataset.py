"""
数据集加载，并对预处理结果进行序列化处理
"""
import sys

from fastNLP.core import Const
from fastNLP.io import PeopleDailyNERLoader
from fastNLP.io import PeopleDailyPipe

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.core.utils import print_data_bundle  # noqa
from myClue.tools.serialize import save_serialize_obj  # noqa


if __name__ == "__main__":
    train_file_config = {
        'train': './data/weibo_NER/train.conll',
        'dev': './data/weibo_NER/dev.conll',
        'test': './data/weibo_NER/test.conll',
    }
    train_data_bundle_pkl_file = './data/weibo_NER/train_data_bundle.pkl'
    logger.info('数据加载')
    data_loader = PeopleDailyNERLoader()
    data_bundle = data_loader.load(train_file_config)
    print_data_bundle(data_bundle)
    logger.info('数据预处理')
    data_pipe = PeopleDailyPipe()
    data_bundle = data_pipe.process(data_bundle)
    data_bundle.rename_field(field_name=Const.CHAR_INPUT, new_field_name=Const.INPUT, ignore_miss_dataset=True, rename_vocab=True)
    print_data_bundle(data_bundle)
    save_serialize_obj(data_bundle, train_data_bundle_pkl_file)
    logger.info('数据预处理后进行序列化：{}'.format(train_data_bundle_pkl_file))
