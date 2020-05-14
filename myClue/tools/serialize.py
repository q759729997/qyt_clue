"""
    module(serialize) - python序列化模块.

    Main members:

        # save_serialize_obj - pickle保存.
        # load_serialize_obj - pickle加载.
"""
import codecs
import pickle


def save_serialize_obj(obj, filename):
    """ pickle保存.

    @params:
        obj - 待保存的对象.
        filename - 路径,文件名建议使用.pkl后缀.
    """
    with codecs.open(filename, 'wb') as fw:
        pickle.dump(obj, fw)


def load_serialize_obj(filename):
    """ pickle加载.

    @return:
        On success - 文件内保存的对象.
    """
    with codecs.open(filename, 'rb') as fread:
        return pickle.load(fread)
