"""
    module(callback) - 模型训练过程中的回调函数.

    Main members:

        # EarlyStopCallback - 早停止回调类.
"""
from fastNLP import Callback
from fastNLP import EarlyStopError

from myClue.core import logger


class EarlyStopCallback(Callback):
    """ 早停止回调类.多少个epoch没有变好就停止训练，相关类 :class:`~fastNLP.core.callback.EarlyStopError`
    """

    def __init__(self, patience):
        """
        :param int patience: epoch的数量
        """
        super().__init__()
        self.patience = patience
        self.wait = 0
        self.epoch_no = 1

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        """
        每次执行验证集的evaluation后会调用。

        :param Dict[str: Dict[str: float]] eval_result: , evaluation的结果。一个例子为{'AccuracyMetric':{'acc':1.0}}，即
            传入的dict是有两层，第一层是metric的名称，第二层是metric的具体指标。
        :param str metric_key: 初始化Trainer时传入的metric_key。
        :param torch.Optimizer optimizer: Trainer中使用的优化器。
        :param bool is_better_eval: 当前dev结果是否比之前的好。
        :return:
        """
        logger.warning('======epoch : {} , early stopping : {}/{}======'.format(self.epoch_no, self.wait, self.patience))
        metric_value = list(eval_result.values())[0].get(metric_key, None)
        logger.info('metric_key : {}, metric_value : {}'.format(metric_key, metric_value))
        logger.info('eval_result : \n{}'.format(eval_result))
        self.epoch_no += 1
        if not is_better_eval:
            # current result is getting worse
            if self.wait == self.patience:
                logger.info('reach early stopping patience, stop training.')
                raise EarlyStopError("Early stopping raised.")
            else:
                self.wait += 1
        else:
            self.wait = 0

    def on_exception(self, exception):
        if isinstance(exception, EarlyStopError):
            logger.info("Early Stopping triggered in epoch {}!".format(self.epoch))
        else:
            raise exception  # 抛出陌生Error
