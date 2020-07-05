import sys
import os
import time

import torch

from fastNLP.core import Const
from fastNLP.embeddings import StaticEmbedding
from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
from torch.optim import Adam

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.core.utils import print_data_bundle  # noqa
from myClue.tools.serialize import save_serialize_obj  # noqa
from myClue.loader.classification import THUCNewsLoader  # noqa
from myClue.pipe.classification import THUCNewsPipe  # noqa
from myClue.core.metrics import ClassifyFPreRecMetric  # noqa
from myClue.core.callback import EarlyStopCallback  # noqa
from myClue.tools.serialize import save_serialize_obj  # noqa
from myClue.tools.file import init_file_path  # noqa
from myClue.models import CNNText  # noqa


if __name__ == "__main__":
    train_file_config = {
        'train': './data/UCAS_NLP_TC/example.txt',
        'dev': './data/UCAS_NLP_TC/example.txt',
        'test': './data/UCAS_NLP_TC/example.txt',
    }
    logger.info('数据加载')
    data_loader = THUCNewsLoader()
    data_bundle = data_loader.load(train_file_config)
    print_data_bundle(data_bundle)
    logger.info('数据预处理')
    data_pipe = THUCNewsPipe()
    data_bundle = data_pipe.process(data_bundle)
    data_bundle.rename_field(field_name=Const.CHAR_INPUT, new_field_name=Const.INPUT, ignore_miss_dataset=True, rename_vocab=True)
    print_data_bundle(data_bundle)
    model_path = './data/UCAS_NLP_TC/model_textcnn_topk'
    init_file_path(model_path)
    logger.add_file_handler(os.path.join(model_path, 'log_{}.txt'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))))  # 日志写入文件
    char_vocab_pkl_file = os.path.join(model_path, 'vocab_char.pkl')
    target_vocab_pkl_file = os.path.join(model_path, 'target_char.pkl')
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
    # metric = AccuracyMetric()
    metric = ClassifyFPreRecMetric(tag_vocab=data_bundle.get_vocab(Const.TARGET), only_gross=False)  # 若only_gross=False, 即还会返回各个label的metric统计值
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果有gpu的话在gpu上运行，训练速度会更快
    logger.info('device:{}'.format(device))
    batch_size = 32
    n_epochs = 5
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
