"""
测试集预测
"""
import os
import sys
import codecs

import torch

from fastNLP import Const
from fastNLP import DataSet
from fastNLP.core.predictor import Predictor

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.tools.serialize import load_serialize_obj  # noqa
from myClue.tools.file import read_file_texts  # noqa


if __name__ == "__main__":
    model_path = './data/tnews_public/model_textcnn'
    char_vocab_pkl_file = os.path.join(model_path, 'vocab_char.pkl')
    target_vocab_pkl_file = os.path.join(model_path, 'target_char.pkl')
    model_name = os.path.join(model_path, 'best_CNNText_f_2020-05-14-23-33-55')
    predict_output_file_name = os.path.join(model_path, 'pred_2020-05-14-23-33-55.txt')
    model = torch.load(model_name)
    model.eval()
    logger.info('模型加载完毕:\n{}'.format(model))
    logger.warn('获取词典')
    char_vocab = load_serialize_obj(char_vocab_pkl_file)
    logger.info('char_vocab:{}'.format(char_vocab))
    target_vocab = load_serialize_obj(target_vocab_pkl_file)
    logger.info('target_vocab:{}'.format(target_vocab))
    logger.warn('加载测试数据')
    test_data_file_name = './data/tnews_public/test.txt'
    test_data_texts = read_file_texts(test_data_file_name)
    test_data = [list(text) for text in test_data_texts]
    logger.info('test_data len:{}'.format(len(test_data)))
    logger.warn('输入数据预处理')
    dataset = DataSet({Const.INPUT: test_data})
    dataset.add_seq_len(field_name=Const.INPUT)
    dataset.set_input(Const.INPUT, Const.INPUT_LEN)
    char_vocab.index_dataset(dataset, field_name=Const.INPUT)
    logger.info('处理后dataset：\n{}'.format(dataset[:5]))
    predictor = Predictor(model)
    # features = [Const.INPUT, Const.INPUT_LEN]
    batch_output = predictor.predict(data=dataset, seq_len_field_name=Const.INPUT_LEN)
    # logger.info('batch_output : {}'.format(batch_output))
    pred_results = batch_output.get('pred')
    logger.info('pred results:{}'.format(pred_results[:5]))
    with codecs.open(predict_output_file_name, mode='w', encoding='utf8') as fw:
        for label_id, text in zip(pred_results, test_data_texts):
            label = target_vocab.to_word(label_id)
            fw.write('{}\t{}\n'.format(label, text))
    logger.warn('预测结果输出：{}'.format(predict_output_file_name))
