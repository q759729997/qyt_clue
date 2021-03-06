"""
    module(file) - 文件与文件夹处理模块.

    Main members:

        # get_file_names_recursion - 递归读取输入路径下的所有文件，file_names会递归更新.
        # read_file_texts - 读取文件内容.
        # read_file_iter - 读取文件内容，使用迭代器返回.
        # read_json_file_iter - 读取json文件内容，使用迭代器返回.
        # init_file_path - 文件路径初始化,若文件夹不存在则进行创建.
"""
__all__ = [
    'get_file_names_recursion',
    'read_file_texts',
    'read_file_iter',
    'read_json_file_iter',
    'init_file_path'
]
import codecs
import json
import os
import pathlib


def get_file_names_recursion(path, file_names):
    """ 递归读取输入路径下的所有文件，file_names会递归更新.

        @params:
            path - 待递归检索的文件夹路径.
            file_names - 待输出结果的文件名列表.

        @return:
            On success - 无返回值，文件输出至file_names中.
    """
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            get_file_names_recursion(file_path, file_names)
        else:
            file_names.append(file_path)


def read_file_texts(file_name, keep_original=False):
    """ 读取文件内容.

        @params:
            file_name - 文件路径.
            keep_original - 保持文件内容不变，默认会去掉空行以及每一行的左右空格.

        @return:
            On success - 文件内容列表，元素为每行.
    """
    with codecs.open(file_name, mode='r', encoding='utf8') as fr:
        texts = list()
        if keep_original:
            for line in fr:
                texts.append(line)
        else:
            # 去掉空行以及每一行的左右空格
            for line in fr:
                line = line.strip()
                if len(line) > 0:
                    texts.append(line)
        return texts


def read_file_iter(file_name, keep_original=False):
    """ 读取文件内容，使用迭代器返回.

        @params:
            file_name - 文件路径.
            keep_original - 保持文件内容不变，默认会去掉空行以及每一行的左右空格.

        @return:
            On success - 文件内容迭代器.
    """
    with codecs.open(file_name, mode='r', encoding='utf8') as fr:
        if keep_original:
            for line in fr:
                yield line
        else:
            # 去掉空行以及每一行的左右空格
            for line in fr:
                line = line.strip()
                if len(line) > 0:
                    yield line


def read_json_file_iter(file_name):
    """ 读取json文件内容，使用迭代器返回.

        @params:
            file_name - json文件名.

        @return:
            On success - 数据迭代器.
            On failure - 错误信息.
    """
    with codecs.open(file_name, mode='r', encoding='utf8') as fr:
        for line in fr:
            line = line.strip()
            if len(line) > 0:
                yield json.loads(line)


def init_file_path(file_path):
    """ 文件路径初始化,若文件夹不存在则进行创建.

        @params:
            file_path - 文件路径.
    """
    pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
