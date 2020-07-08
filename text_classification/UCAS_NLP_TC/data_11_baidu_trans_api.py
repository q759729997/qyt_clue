"""
百度翻译API
"""
import time
import os
import sys
import codecs
import json
import traceback
from tqdm import tqdm
import http.client
import hashlib
import urllib
import random

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级
from data import baidu_config  # noqa

""" 你的 APPID AK SK """
appid = baidu_config.trans_APP_ID  # 填写你的appid
secretKey = baidu_config.trans_SECRET_KEY  # 填写你的密钥

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.tools.file import read_file_texts  # noqa
from myClue.tools.file import init_file_path  # noqa


def get_baidu_trans(text):
    for i in range(20):
        try:
            # text = text.encode('gbk', errors='ignore').decode('gbk', errors='ignore')  # 去掉GBK不识别的字符串，该接口接收GBK格式
            # 待翻译文本（q）需为UTF-8编码
            if len(text) == 0:
                return []
            httpClient = None
            myurl = '/api/trans/vip/translate'
            fromLang = 'auto'   # 原文语种
            toLang = 'en'   # 译文语种
            salt = random.randint(32768, 65536)
            # 单次翻译文本长度限定为6000字节以内（汉字约为2000个字符）
            sign = appid + text + str(salt) + secretKey
            sign = hashlib.md5(sign.encode()).hexdigest()
            myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(text) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign
            httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
            httpClient.request('GET', myurl)
            # response是HTTPResponse对象
            response = httpClient.getresponse()
            result_all = response.read().decode("utf-8")
            result = json.loads(result_all)
            if 'trans_result' in result:
                return result['trans_result']
            else:
                continue
        except Exception as e:
            print('text:{}, i:{}, exception：{}'.format(text, i, e))
            traceback.print_exc()
            time.sleep(2)
    return []


if __name__ == "__main__":
    train_file_config = {
        'dev': './data/UCAS_NLP_TC/devdata.txt',
        'test': './data/UCAS_NLP_TC/testdata.txt',
        'train': './data/UCAS_NLP_TC/traindata.txt',
    }
    output_path = './data/UCAS_NLP_TC/data_baidu_trans'
    init_file_path(output_path)
    for file_label, file_name in train_file_config.items():
        logger.info('开始处理：{}'.format(file_label))
        texts = read_file_texts(file_name)
        output_file_name = os.path.join(output_path, '{}_trans.json'.format(file_label))
        with codecs.open(output_file_name, mode='w', encoding='utf8') as fw:
            for text in tqdm(texts):
                label, news_content = text.split('\t')
                news_content = news_content.replace(' ', '')
                step = 500
                splited_texts = [news_content[i:i+step] for i in range(0, len(news_content), step)]
                trans_results = list()
                for splited_text in splited_texts:
                    trans_results.extend(get_baidu_trans(splited_text))
                row_json = {
                    'label': label,
                    'trans_results': trans_results,
                    'news_content': news_content
                }
                fw.write('{}\n'.format(json.dumps(row_json, ensure_ascii=False)))
                time.sleep(1)
                # break
