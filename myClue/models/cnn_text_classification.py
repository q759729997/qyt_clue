r"""
.. todo::
    doc
"""

__all__ = [
    "CNNText"
]

import torch
import torch.nn as nn

from fastNLP.core.const import Const as C
from fastNLP.core.utils import seq_len_to_mask
from fastNLP.embeddings import embedding
from fastNLP.modules import encoder


class CNNText(torch.nn.Module):
    r"""
    使用CNN进行文本分类的模型
    'Yoon Kim. 2014. Convolution Neural Networks for Sentence Classification.'

    """

    def __init__(self, embed,
                 num_classes,
                 kernel_nums=(30, 40, 50),
                 kernel_sizes=(1, 3, 5),
                 dropout=0.5):
        r"""

        :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray embed: Embedding的大小(传入tuple(int, int),
            第一个int为vocab_zie, 第二个int为embed_dim); 如果为Tensor, Embedding, ndarray等则直接使用该值初始化Embedding
        :param int num_classes: 一共有多少类
        :param int,tuple(int) kernel_sizes: 输出channel的kernel大小。
        :param float dropout: Dropout的大小
        """
        super(CNNText, self).__init__()

        # no support for pre-trained embedding currently
        self.embed = embedding.Embedding(embed)
        self.conv_pool = encoder.ConvMaxpool(
            in_channels=self.embed.embedding_dim,
            out_channels=kernel_nums,
            kernel_sizes=kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(kernel_nums), num_classes)

    def forward(self, words, seq_len=None):
        r"""

        :param torch.LongTensor words: [batch_size, seq_len]，句子中word的index
        :param torch.LongTensor seq_len:  [batch,] 每个句子的长度
        :return output: dict of torch.LongTensor, [batch_size, num_classes]
        """
        x = self.embed(words)  # [N,L] -> [N,L,C]
        if seq_len is not None:
            mask = seq_len_to_mask(seq_len)
            x = self.conv_pool(x, mask)
        else:
            x = self.conv_pool(x)  # [N,L,C] -> [N,C]
        x = self.dropout(x)
        x = self.fc(x)  # [N,C] -> [N, N_class]
        return {C.OUTPUT: x}

    def predict(self, words, seq_len=None):
        r"""
        :param torch.LongTensor words: [batch_size, seq_len]，句子中word的index
        :param torch.LongTensor seq_len:  [batch,] 每个句子的长度

        :return predict: dict of torch.LongTensor, [batch_size, ]
        """
        output = self.forward(words, seq_len)
        logits = output[C.OUTPUT]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # print('logits:{}'.format(logits))
        # print('probs:{}'.format(probs))
        predict_prob, predict = probs.max(dim=1)
        k = probs.shape[-1]
        top_k_prob, top_k_predict = probs.topk(k=k, dim=1)
        return {C.OUTPUT: predict, 'prob': predict_prob, 'top_k_predict': top_k_predict, 'top_k_prob': top_k_prob}
