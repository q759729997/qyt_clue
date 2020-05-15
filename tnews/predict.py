"""
测试集预测
"""
import os
import sys
import json
import codecs
from collections import OrderedDict

import torch

from fastNLP import Const
from fastNLP import DataSet
from fastNLP.core.predictor import Predictor

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.tools.serialize import load_serialize_obj  # noqa
from myClue.tools.file import read_json_file_iter  # noqa
from myClue.tools.text import remove_blank  # noqa


if __name__ == "__main__":
    """
    预测示例输出结果：{"id": 0, "label": "102", "label_desc": "news_entertainment"}
    """
    model_path = './data/tnews_public/model_textcnn'
    test_data_json_file_name = './data/tnews_public/test.json'
    label_json_file_name = './data/tnews_public/labels.json'
    char_vocab_pkl_file = os.path.join(model_path, 'vocab_char.pkl')
    target_vocab_pkl_file = os.path.join(model_path, 'target_char.pkl')
    model_name = os.path.join(model_path, 'best_CNNText_f_2020-05-14-23-33-55')
    predict_output_json_file_name = os.path.join(model_path, 'pred_2020-05-14-23-33-55.json')
    predict_output_file_name = os.path.join(model_path, 'pred_2020-05-14-23-33-55.txt')
    logger.warn('加载标签映射关系')
    json_file_iter = read_json_file_iter(label_json_file_name)
    label_link_dict = dict()
    for row_json in json_file_iter:
        label_link_dict[row_json['label_desc']] = row_json['label']
    logger.info(label_link_dict)
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
    json_file_iter = read_json_file_iter(test_data_json_file_name)
    predictor = Predictor(model)
    with codecs.open(predict_output_json_file_name, mode='w', encoding='utf8') as fw_json, codecs.open(predict_output_file_name, mode='w', encoding='utf8') as fw:
        for i, row_json in enumerate(json_file_iter):
            logger.info('predict row:{}'.format(i))
            sentence = row_json.get('sentence', '')
            keywords = row_json.get('keywords', '')
            text = remove_blank('{}{}'.format(sentence, keywords))
            input_data = []
            test_data = [list(text)]
            # logger.info('test_data len:{}'.format(len(test_data)))
            # logger.warn('输入数据预处理')
            dataset = DataSet({Const.INPUT: test_data})
            dataset.add_seq_len(field_name=Const.INPUT)
            dataset.set_input(Const.INPUT, Const.INPUT_LEN)
            char_vocab.index_dataset(dataset, field_name=Const.INPUT)
            # logger.info('处理后dataset：\n{}'.format(dataset[:5]))
            # features = [Const.INPUT, Const.INPUT_LEN]
            batch_output = predictor.predict(data=dataset, seq_len_field_name=Const.INPUT_LEN)
            # logger.info('batch_output : {}'.format(batch_output))
            pred_results = batch_output.get('pred')
            # logger.info('pred results:{}'.format(pred_results[:5]))
            label_id = pred_results[0]
            label_desc = target_vocab.to_word(label_id)
            # 组装成所需格式
            row_data = OrderedDict()
            row_data['id'] = row_json['id']
            row_data['label'] = label_link_dict[label_desc]
            row_data['label_desc'] = label_desc
            fw_json.write('{}\n'.format(json.dumps(row_data, ensure_ascii=False)))
            fw.write('{}\t{}\n'.format(label_desc, text))
            if i > 5:
                break
