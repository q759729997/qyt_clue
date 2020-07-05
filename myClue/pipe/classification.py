import traceback

from fastNLP.core.const import Const
from fastNLP.io.data_bundle import DataBundle
from fastNLP.io.loader import THUCNewsLoader
from fastNLP.io.pipe import Pipe
from fastNLP.io.pipe.utils import get_tokenizer
from fastNLP.io.pipe.utils import _indexize

from myClue.core import logger


def get_data_bundle_tags(data_bundle: DataBundle):
    """ 根据dataBundle获取tags.

    @params:
        data_bundle - DataBundle数据集.

    @return:
        On success - 数据标签的tag列表.
    """
    try:
        dataset = data_bundle.get_dataset('train')
        target_names = dataset.get_field(Const.TARGET).content
        target_names = list(set(target_names))
    except Exception:
        traceback.print_exc()
        logger.error('缺少train数据集')
        raise Exception('缺少train数据集')
    target_names = list(set(target_names))
    target_names.sort()
    return target_names


class _CLSPipe(Pipe):
    """
    分类问题的基类，负责对classification的数据进行tokenize操作。默认是对raw_words列操作，然后生成words列

    """
    def __init__(self, tokenizer: str = 'spacy', lang='en'):

        self.tokenizer = get_tokenizer(tokenizer, lang=lang)

    def _tokenize(self, data_bundle, field_name=Const.INPUT, new_field_name=None):
        """
        将DataBundle中的数据进行tokenize

        :param DataBundle data_bundle:
        :param str field_name:
        :param str new_field_name:
        :return: 传入的DataBundle对象
        """
        new_field_name = new_field_name or field_name
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self.tokenizer, field_name=field_name, new_field_name=new_field_name)

        return data_bundle

    def _granularize(self, data_bundle, tag_map):
        """
        该函数对data_bundle中'target'列中的内容进行转换。

        :param data_bundle:
        :param dict tag_map: 将target列中的tag做以下的映射，比如{"0":0, "1":0, "3":1, "4":1}, 则会删除target为"2"的instance，
            且将"1"认为是第0类。
        :return: 传入的data_bundle
        """
        for name in list(data_bundle.datasets.keys()):
            dataset = data_bundle.get_dataset(name)
            dataset.apply_field(lambda target: tag_map.get(target, -100), field_name=Const.TARGET,
                                new_field_name=Const.TARGET)
            dataset.drop(lambda ins: ins[Const.TARGET] == -100)
            data_bundle.set_dataset(dataset, name)
        return data_bundle


class THUCNewsPipe(_CLSPipe):
    """
    处理之后的DataSet有以下的结构

    .. csv-table::
        :header: "raw_chars", "target", "chars", "seq_len"

        "马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道...", 0, "[409, 1197, 2146, 213, ...]", 746
        "..."

    其中chars, seq_len是input，target是target
    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+-----------+--------+-------+---------+
        | field_names | raw_chars | target | chars | seq_len |
        +-------------+-----------+--------+-------+---------+
        |   is_input  |   False   |  True  |  True |   True  |
        |  is_target  |   False   |  True  | False |  False  |
        | ignore_type |           | False  | False |  False  |
        |  pad_value  |           |   0    |   0   |    0    |
        +-------------+-----------+--------+-------+---------+

    :param bool bigrams: 是否增加一列bigrams. bigrams的构成是['复', '旦', '大', '学', ...]->["复旦", "旦大", ...]。如果
        设置为True，返回的DataSet将有一列名为bigrams, 且已经转换为了index并设置为input，对应的vocab可以通过
        data_bundle.get_vocab('bigrams')获取.
    :param bool trigrams: 是否增加一列trigrams. trigrams的构成是 ['复', '旦', '大', '学', ...]->["复旦大", "旦大学", ...]
        。如果设置为True，返回的DataSet将有一列名为trigrams, 且已经转换为了index并设置为input，对应的vocab可以通过
        data_bundle.get_vocab('trigrams')获取.
    """

    def __init__(self, bigrams=False, trigrams=False, tokenizer='char'):
        super().__init__(tokenizer='cn-char', lang='cn')

        self.bigrams = bigrams
        self.trigrams = trigrams
        self.tokenizer = tokenizer

    def _chracter_split(self, sent):
        return list(sent)

    def _white_space_split(self, sent):
        return sent.split(' ')

    def _raw_split(self, sent):
        return sent.split()

    def _tokenize(self, data_bundle, field_name=Const.INPUT, new_field_name=None):
        new_field_name = new_field_name or field_name
        if self.tokenizer == 'char':
            for name, dataset in data_bundle.datasets.items():
                dataset.apply_field(self._chracter_split, field_name=field_name, new_field_name=new_field_name)
                dataset.copy_field(field_name=new_field_name, new_field_name='raw_words')
        elif self.tokenizer == 'white_space':
            for name, dataset in data_bundle.datasets.items():
                dataset.apply_field(self._white_space_split, field_name=field_name, new_field_name=new_field_name)
                dataset.copy_field(field_name=new_field_name, new_field_name='raw_words')
        return data_bundle

    def process(self, data_bundle: DataBundle):
        """
        可处理的DataSet应具备如下的field

        .. csv-table::
            :header: "raw_words", "target"

            "马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道 ... ", "体育"
            "...", "..."

        :param data_bundle:
        :return:
        """
        # 根据granularity设置tag
        # 由原来的固定tagmap，修改为根据数据集获取tagmap
        targets_vocabs = get_data_bundle_tags(data_bundle)
        self.tag_map = {tag_name: tag_name for tag_name in targets_vocabs}
        data_bundle = self._granularize(data_bundle=data_bundle, tag_map=self.tag_map)
        # clean,lower

        # CWS(tokenize)
        data_bundle = self._tokenize(data_bundle=data_bundle, field_name='raw_chars', new_field_name='chars')
        input_field_names = [Const.CHAR_INPUT]

        # n-grams
        if self.bigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])],
                                    field_name=Const.CHAR_INPUT, new_field_name='bigrams')
            input_field_names.append('bigrams')
        if self.trigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(lambda chars: [c1 + c2 + c3 for c1, c2, c3 in
                                                   zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)],
                                    field_name=Const.CHAR_INPUT, new_field_name='trigrams')
            input_field_names.append('trigrams')

        # index
        data_bundle = _indexize(data_bundle=data_bundle, input_field_names=Const.CHAR_INPUT)
        # add length
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(field_name=Const.CHAR_INPUT, new_field_name=Const.INPUT_LEN)

        # input_fields包含的字段名称
        # input_fields = [Const.TARGET, Const.INPUT_LEN] + input_field_names
        input_fields = [Const.INPUT_LEN] + input_field_names
        target_fields = [Const.TARGET]

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths=None):
        """
        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.Loader` 的load函数。
        :return: DataBundle
        """
        data_loader = THUCNewsLoader()  # 此处需要实例化一个data_loader，否则传入load()的参数为None
        data_bundle = data_loader.load(paths)
        data_bundle = self.process(data_bundle)
        return data_bundle
