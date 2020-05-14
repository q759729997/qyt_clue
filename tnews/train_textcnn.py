""""
TextCNN模型训练
"""
import os
import sys

import torch

from fastNLP.embeddings import StaticEmbedding
from fastNLP.models import CNNText
from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
from torch.optim import Adam
from fastNLP import AccuracyMetric
from fastNLP import Tester

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.tools.serialize import load_serialize_obj  # noqa
from myClue.tools.serialize import save_serialize_obj  # noqa
from myClue.tools.file import init_file_path  # noqa


if __name__ == "__main__":
    train_data_bundle_pkl_file = './data/tnews_public/train_data_bundle_char.pkl'
    model_path = './data/tnews_public/model_textcnn'
    char_vocab_pkl_file = os.path.join(model_path, 'vocab_char.pkl')
    target_vocab_pkl_file = os.path.join(model_path, 'target_char.pkl')
    init_file_path(model_path)
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
    word2vec_embed = StaticEmbedding(char_vocab, model_dir_or_name='cn-char-fastnlp-100d')
    logger.warn('神经网络模型')
    model = CNNText(word2vec_embed, num_classes=len(target_vocab))
    logger.info(model)
    logger.warn('训练超参数设定')
    loss = CrossEntropyLoss()
    optimizer = Adam([param for param in model.parameters() if param.requires_grad])
    metric = AccuracyMetric()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果有gpu的话在gpu上运行，训练速度会更快
    logger.info('device:{}'.format(device))
    batch_size = 32
    n_epochs = 5
    trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, loss=loss,
                      optimizer=optimizer, batch_size=batch_size, n_epochs=n_epochs, dev_data=data_bundle.get_dataset('dev'),
                      metrics=metric, device=device)
    logger.warn('开始训练')
    trainer.train()
    logger.warn('训练后评估')
    tester = Tester(data=data_bundle.get_dataset('test'), model=model, metrics=metric, batch_size=64, device=device)
    tester.test()
