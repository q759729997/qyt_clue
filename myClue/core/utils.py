from fastNLP.io.data_bundle import DataBundle

from myClue.core import logger


def print_data_bundle(data_bundle: DataBundle, title: str = None):
    """ 打印输出data_bundle的信息.

    @params:
        data_bundle - 数据集DataBundle.
        title - 打印输出的标题信息.
    """
    if title:
        logger.warning(title)
    for name, dataset in data_bundle.iter_datasets():
        logger.info('dataset name : {}'.format(name))
        logger.info('dataset len : {}'.format(len(dataset)))
        logger.info('dataset example : ')
        logger.info('\n{}'.format(dataset[:5]))
        logger.info('dataset 输出各个field的被设置成input和target的情况 : ')
        logger.info('\n{}'.format(dataset.print_field_meta()))
