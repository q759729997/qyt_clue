""""
TextCNN模型训练
"""
import os
import sys
import time

import torch

from fastNLP import Const
from fastNLP.embeddings import BertEmbedding
from fastNLP.models import BiLSTMCRF
from fastNLP import SpanFPreRecMetric
from fastNLP import Trainer
from fastNLP import LossInForward
from torch.optim import Adam
from fastNLP import Tester

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.core.callback import EarlyStopCallback  # noqa
from myClue.tools.serialize import load_serialize_obj  # noqa
from myClue.tools.serialize import save_serialize_obj  # noqa
from myClue.tools.file import init_file_path  # noqa


if __name__ == "__main__":
    train_data_bundle_pkl_file = './data/weibo_NER/train_data_bundle.pkl'
    model_path = './data/weibo_NER/model_bilstm_crf_bert_embed'
    init_file_path(model_path)
    logger.add_file_handler(os.path.join(model_path, 'log_{}.txt'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))))  # 日志写入文件
    char_vocab_pkl_file = os.path.join(model_path, 'vocab_char.pkl')
    target_vocab_pkl_file = os.path.join(model_path, 'target_char.pkl')
    logger.warn('加载数据集')
    data_bundle = load_serialize_obj(train_data_bundle_pkl_file)
    logger.warn('获取词典')
    char_vocab = data_bundle.get_vocab('words')
    logger.info('char_vocab:{}'.format(char_vocab))
    target_vocab = data_bundle.get_vocab('target')
    logger.info('target_vocab:{}'.format(target_vocab))
    save_serialize_obj(char_vocab, char_vocab_pkl_file)
    save_serialize_obj(target_vocab, target_vocab_pkl_file)
    logger.info('词典序列化:{}'.format(char_vocab_pkl_file))
    logger.warn('选择预训练词向量')
    bert_embed = BertEmbedding(vocab=char_vocab, model_dir_or_name='cn-wwm', requires_grad=False)
    logger.warn('神经网络模型')
    model = BiLSTMCRF(embed=bert_embed, num_classes=len(target_vocab), num_layers=1, hidden_size=200, dropout=0.5,
                      target_vocab=target_vocab)
    logger.info(model)
    logger.warn('训练超参数设定')
    loss = LossInForward()
    optimizer = Adam([param for param in model.parameters() if param.requires_grad])
    # metric = AccuracyMetric()
    metric = SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab(Const.TARGET), only_gross=False)  # 若only_gross=False, 即还会返回各个label的metric统计值
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果有gpu的话在gpu上运行，训练速度会更快
    logger.info('device:{}'.format(device))
    batch_size = 32
    n_epochs = 10
    early_stopping = 10
    trainer = Trainer(
        save_path=model_path,
        train_data=data_bundle.get_dataset('train'),
        model=model,
        loss=loss,
        optimizer=optimizer,
        batch_size=batch_size,
        n_epochs=n_epochs,
        dev_data=data_bundle.get_dataset('dev'),
        metrics=metric,
        metric_key='f',
        device=device,
        callbacks=[EarlyStopCallback(early_stopping)])
    logger.warn('开始训练')
    trainer.train()
    logger.warn('训练后评估')
    tester = Tester(data=data_bundle.get_dataset('test'), model=model, metrics=metric, batch_size=64, device=device)
    test_result = tester.test()
    logger.info('评估结果：\n{}'.format(test_result))
