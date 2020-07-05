import torch
import os
import sys
import codecs

from fastNLP import Const
from fastNLP import DataSet
from fastNLP.core.predictor import Predictor

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.tools.serialize import load_serialize_obj  # noqa
from myClue.tools.file import read_json_file_iter  # noqa


if __name__ == "__main__":
    model_path = './data/UCAS_NLP_TC/model_textcnn_topk'
    char_vocab_pkl_file = os.path.join(model_path, 'vocab_char.pkl')
    target_vocab_pkl_file = os.path.join(model_path, 'target_char.pkl')
    model_name = os.path.join(model_path, 'best_CNNText_f_2020-07-04-23-18-38-341248')
    logger.warn('开始加载模型')
    model = torch.load(model_name)
    model.eval()
    logger.info('模型加载完毕:\n{}'.format(model))
    logger.warn('获取词典')
    char_vocab = load_serialize_obj(char_vocab_pkl_file)
    logger.info('char_vocab:{}'.format(char_vocab))
    target_vocab = load_serialize_obj(target_vocab_pkl_file)
    logger.info('target_vocab:{}'.format(target_vocab))
    logger.warn('加载测试数据')
    text = "世界贸易组织（WTO）17日对美国进行贸易政策审议。在当天的会议上，包括中国、欧"
    test_data = [list(text)]
    dataset = DataSet({Const.INPUT: test_data})
    dataset.add_seq_len(field_name=Const.INPUT)
    dataset.set_input(Const.INPUT, Const.INPUT_LEN)
    char_vocab.index_dataset(dataset, field_name=Const.INPUT)
    predictor = Predictor(model)
    batch_output = predictor.predict(data=dataset, seq_len_field_name=Const.INPUT_LEN)
    logger.info('batch_output : {}'.format(batch_output))
    pred_results = batch_output.get('pred')
    logger.info('pred results:{}'.format(pred_results[:5]))
    label_id = pred_results[0]
    pred_label = target_vocab.to_word(label_id)
    logger.info('label_id：{}, pred_label : {}'.format(label_id, pred_label))
    # topk概率
    top_k_predict = batch_output.get('top_k_predict')[0]
    top_k_prob = batch_output.get('top_k_prob')[0]
    predict_results = list()
    for label_id, prob in zip(top_k_predict, top_k_prob):
        predict_results.append((target_vocab.to_word(label_id), prob))
    print(predict_results)
