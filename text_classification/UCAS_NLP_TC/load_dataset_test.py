"""
数据集加载测试
"""
import sys

from fastNLP.core import Const

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.core.utils import print_data_bundle  # noqa
from myClue.loader.classification import THUCNewsLoader  # noqa
from myClue.pipe.classification import THUCNewsPipe  # noqa
from myClue.tools.serialize import save_serialize_obj  # noqa


if __name__ == "__main__":
    train_file_config = {
        'train': './data/UCAS_NLP_TC/data_01_shuffle/traindata.txt',
        'dev': './data/UCAS_NLP_TC/data_01_shuffle/devdata.txt',
    }
    logger.info('数据加载')
    data_loader = THUCNewsLoader()
    data_bundle = data_loader.load(train_file_config)
    print_data_bundle(data_bundle)
    logger.info('数据预处理')
    data_pipe = THUCNewsPipe(tokenizer='white_space')
    data_bundle = data_pipe.process(data_bundle)
    data_bundle.rename_field(field_name=Const.CHAR_INPUT, new_field_name=Const.INPUT, ignore_miss_dataset=True, rename_vocab=True)
    print_data_bundle(data_bundle)
