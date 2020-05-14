"""
    module(text) - 文本处理模块.

    Main members:

        # remove_blank - 匹配任意空白字符[\t\n\r\f]，替换为指定字符.
        # get_json_dumps - Python复杂数据类型转json字符串，保持中文不改变.
        # decode_bytes - 字节流解码为字符串文本.
"""
__all__ = [
    'remove_blank',
    'get_json_dumps',
    'decode_bytes'
]

import chardet
import json
import re


def remove_blank(text, replace_text=''):
    """ 匹配任意空白字符[\t\n\r\f]，替换为指定字符.

        @params:
            text - 字符串文本.
            replace_text - 需要替换的文本.

        @return:
            On success - 处理之后的文本.
    """
    return re.sub(r'\s+', replace_text, text)


def get_json_dumps(json_obj) -> str:
    """ Python复杂数据类型转json字符串，保持中文不改变.

        @params:
            json_obj - 待转换字符串的数据.

        @return:
            On success - json字符串.
    """
    return json.dumps(json_obj, ensure_ascii=False)


def decode_bytes(byte_stream, return_encoding_type=False) -> str:
    """ 字节流解码为字符串文本.

        @params:
            byte_stream - 字节流文本.
            return_encoding_type - 返回解码类型.

        @return:
            On success - 解码后的字符串文本;return_encoding_type为真时返回元组(字符串文本, encoding_type).
    """
    encoding_type = chardet.detect(byte_stream)['encoding']
    text = byte_stream.decode(encoding_type)
    if return_encoding_type:
        return text, encoding_type
    return text
