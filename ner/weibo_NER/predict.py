import os
import sys
import codecs
import copy

import torch
from fastNLP.core import Const
from fastNLP.core.predictor import Predictor
from fastNLP.io import PeopleDailyNERLoader

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.core.utils import print_data_bundle  # noqa
from myClue.tools.serialize import load_serialize_obj  # noqa


if __name__ == "__main__":
    model_path = './data/weibo_NER/model_bilstm_crf_random_embed'
    model_file = os.path.join(model_path, 'best_BiLSTMCRF_f_2020-05-20-11-08-16-221138')
    train_file = './data/weibo_NER/example.conll'
    predict_output_file = './data/weibo_NER/example_BiLSTMCRF_predict.conll'
    char_vocab_pkl_file = os.path.join(model_path, 'vocab_char.pkl')
    target_vocab_pkl_file = os.path.join(model_path, 'target_char.pkl')
    # 加载数据
    data_loader = PeopleDailyNERLoader()
    data_bundle = data_loader.load({'train': train_file})
    print_data_bundle(data_bundle)
    dataset = data_bundle.datasets['train']
    dataset_original = copy.deepcopy(dataset)
    # 加载词表
    char_vocab = load_serialize_obj(char_vocab_pkl_file)
    logger.info('char_vocab:{}'.format(char_vocab))
    target_vocab = load_serialize_obj(target_vocab_pkl_file)
    logger.info('target_vocab:{}'.format(target_vocab))
    # 加载模型
    model = torch.load(model_file)
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info('use cuda')
    model.eval()
    logger.info('模型加载完毕:\n{}'.format(model))
    # 数据预处理
    dataset.rename_field(field_name=Const.RAW_CHAR, new_field_name=Const.INPUT)
    dataset.add_seq_len(field_name=Const.INPUT)
    dataset.set_input(Const.INPUT, Const.INPUT_LEN)
    dataset.set_target(Const.TARGET, Const.INPUT_LEN)
    char_vocab.index_dataset(dataset, field_name=Const.INPUT)
    # 预测
    predictor = Predictor(model)
    predict_output = predictor.predict(data=dataset, seq_len_field_name=Const.INPUT_LEN)
    pred_results = predict_output.get(Const.OUTPUT)
    # 预测结果解码
    with codecs.open(predict_output_file, mode='w', encoding='utf8') as fw:
        for datarow, pred_result in zip(dataset_original, pred_results):
            pred_result = [target_vocab.to_word(pred_item) for pred_item in pred_result]
            row_chars = datarow[Const.RAW_CHAR]
            for char, label in zip(row_chars, pred_result):
                fw.write('{}\t{}\n'.format(char, label))
            fw.write('\n')
        # fw.write('\n')
    logger.info('predict_output_file：{}'.format(predict_output_file))
