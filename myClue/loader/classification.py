import codecs

from fastNLP import DataSet
from fastNLP import Instance
from fastNLP.io import Loader


class THUCNewsLoader(Loader):
    """
    数据集简介：document-level分类任务，新闻10分类
    原始数据内容为：每行一个sample，第一个'\t'之前为target，第一个'\t'之后为raw_words

    Example::

        体育	调查-您如何评价热火客场胜绿军总分3-1夺赛点？...

    读取后的Dataset将具有以下数据结构：

    .. csv-table::
       :header: "raw_words", "target"

       "调查-您如何评价热火客场胜绿军总分3-1夺赛点？...", "体育"
       "...", "..."

    """

    def __init__(self):
        super(THUCNewsLoader, self).__init__()

    def _load(self, path: str = None):
        ds = DataSet()
        with codecs.open(path, mode='r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                if len(line) == 0:
                    continue
                sep_index = line.index('\t')
                raw_chars = line[sep_index + 1:]
                target = line[:sep_index]
                if raw_chars:
                    ds.append(Instance(raw_chars=raw_chars, target=target))
        return ds
