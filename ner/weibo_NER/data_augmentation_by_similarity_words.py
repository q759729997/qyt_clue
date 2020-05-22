import sys
import codecs

from gensim.models import KeyedVectors
from fastNLP.core import Const
from fastNLP.io import PeopleDailyNERLoader

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.core.utils import print_data_bundle  # noqa
from myClue.tools.serialize import load_serialize_obj  # noqa
from myClue.tools.decoder import decode_ner_tags  # noqa


if __name__ == "__main__":
    """通过word2vec中最相似的词进行数据增强"""
    # train_file = './data/weibo_NER/example.conll'
    # augmentation_file = './data/weibo_NER/example_augmentation.conll'
    train_file = './data/weibo_NER/train.conll'
    augmentation_file = './data/weibo_NER/train_augmentation.conll'
    # train_file = './data/weibo_NER/example.conll'
    # augmentation_file = './data/weibo_NER/example_augmentation.conll'
    logger.info('加载word2vec')
    word2vec_model_file = './data/embed/sgns.weibo.word/sgns.weibo.word'
    model = KeyedVectors.load_word2vec_format(word2vec_model_file, binary=False)
    logger.info('word2vec加载完毕，测试一下:{}'.format(model.most_similar('扎克伯格', topn=5)))
    # 加载数据
    data_loader = PeopleDailyNERLoader()
    data_bundle = data_loader.load({'train': train_file})
    print_data_bundle(data_bundle)
    dataset = data_bundle.datasets['train']
    # 数据处理
    with codecs.open(augmentation_file, mode='w', encoding='utf8') as fw:
        for row_id, datarow in enumerate(dataset):
            if row_id % 10 == 0:
                print('row_id:{}'.format(row_id))
            row_chars = datarow[Const.RAW_CHAR]
            target = datarow[Const.TARGET]
            ner_tags = decode_ner_tags(row_chars, target)
            if len(ner_tags) == 0:
                continue
            for entity in ner_tags:
                # 匹配相似度
                word = entity['word']
                ne_type = entity['type']
                offset = entity['offset']
                length = entity['length']
                try:
                    similarity_words = model.most_similar(word, topn=1)
                except Exception:  # 处理word不在word2vec的情况
                    continue
                similarity_word, similarity_score = similarity_words[0]
                if word == similarity_word or similarity_score < 0.5:
                    continue
                # 组装结果
                augmentation_chars, augmentation_tags = list(), list()
                # 前缀
                augmentation_chars.extend(row_chars[:offset])
                augmentation_tags.extend(target[:offset])
                # 实体
                augmentation_chars.extend(list(similarity_word))
                for char_index in range(len(similarity_word)):
                    if char_index == 0:
                        augmentation_tags.append('B-{}'.format(ne_type))
                    else:
                        augmentation_tags.append('I-{}'.format(ne_type))
                # 后缀
                augmentation_chars.extend(row_chars[offset+length:])
                augmentation_tags.extend(target[offset+length:])
                # 输出
                for char, label in zip(augmentation_chars, augmentation_tags):
                    fw.write('{}\t{}\n'.format(char, label))
                fw.write('\n')
        # fw.write('\n')
    logger.info('augmentation_file{}'.format(augmentation_file))
