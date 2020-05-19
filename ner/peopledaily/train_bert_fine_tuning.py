""""
TextCNN模型训练
"""
import os
import sys
import time

import torch

from fastNLP import Const
from fastNLP.core.callback import GradientClipCallback
from fastNLP.core.callback import WarmupCallback
from fastNLP import SpanFPreRecMetric
from fastNLP import Trainer
from fastNLP import LossInForward
from fastNLP.core.optimizer import AdamW
from fastNLP.embeddings import BertEmbedding
from fastNLP import Tester

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.core.callback import EarlyStopCallback  # noqa
from myClue.models.bert_crf import BertCRF  # noqa
from myClue.tools.serialize import load_serialize_obj  # noqa
from myClue.tools.serialize import save_serialize_obj  # noqa
from myClue.tools.file import init_file_path  # noqa


if __name__ == "__main__":
    train_data_bundle_pkl_file = './data/peopledaily/train_data_bundle.pkl'
    model_path = './data/peopledaily/model_bert_fine_tuning'
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
    bert_embed = BertEmbedding(vocab=char_vocab, model_dir_or_name='cn-wwm', pool_method='max', requires_grad=True, layers='11', include_cls_sep=False, dropout=0.5, word_dropout=0.01, auto_truncate=True)
    logger.warn('神经网络模型')
    model = BertCRF(bert_embed, tag_vocab=target_vocab, encoding_type='bio')
    logger.info(model)
    logger.warn('训练超参数设定')
    loss = LossInForward()
    optimizer = AdamW([param for param in model.parameters() if param.requires_grad], lr=2e-5)
    # metric = AccuracyMetric()
    metric = SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab(Const.TARGET), only_gross=False)  # 若only_gross=False, 即还会返回各个label的metric统计值
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果有gpu的话在gpu上运行，训练速度会更快
    logger.info('device:{}'.format(device))
    batch_size = 32
    n_epochs = 10
    early_stopping = 10
    callbacks = [GradientClipCallback(clip_type='norm', clip_value=1),
                 WarmupCallback(warmup=0.1, schedule='linear'),
                 EarlyStopCallback(early_stopping)]
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
        callbacks=callbacks)
    logger.warn('开始训练')
    trainer.train()
    logger.warn('训练后评估')
    tester = Tester(data=data_bundle.get_dataset('test'), model=model, metrics=metric, batch_size=64, device=device)
    test_result = tester.test()
    logger.info('评估结果：\n{}'.format(test_result))
