"""
百度词法分析API：
pip install baidu-aip
"""
import time
import os
import sys
import codecs
import json
import traceback
from tqdm import tqdm
from aip import AipNlp

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级
from data import baidu_config  # noqa

""" 你的 APPID AK SK """
APP_ID = baidu_config.APP_ID  # '你的 App ID'
API_KEY = baidu_config.API_KEY  # '你的 Api Key'
SECRET_KEY = baidu_config.SECRET_KEY  # '你的 Secret Key'

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

# text = "百度是一家高科技公司"

""" 调用词法分析 """
# print(client.lexer(text))

import myClue  # noqa
print('myClue module path :{}'.format(myClue.__file__))  # 输出测试模块文件位置
from myClue.core import logger  # noqa
from myClue.tools.file import read_file_texts  # noqa
from myClue.tools.file import init_file_path  # noqa


def get_baidu_cws(text):
    for i in range(20):
        try:
            text = text.encode('gbk', errors='ignore').decode('gbk', errors='ignore')  # 去掉GBK不识别的字符串，该接口接收GBK格式
            cws_result = client.lexer(text)
            if 'items' in cws_result:
                return cws_result['items']
            else:
                continue
        except Exception as e:
            time.sleep(0.5)
            print('text:{}, i:{}, exception：{}'.format(text, i, e))
            traceback.print_exc()
    return []


if __name__ == "__main__":
    train_file_config = {
        'dev': './data/UCAS_NLP_TC/devdata.txt',
        'test': './data/UCAS_NLP_TC/testdata.txt',
        'train': './data/UCAS_NLP_TC/traindata.txt',
    }
    output_path = './data/UCAS_NLP_TC/data_baidu_cws'
    init_file_path(output_path)
    for file_label, file_name in train_file_config.items():
        logger.info('开始处理：{}'.format(file_label))
        texts = read_file_texts(file_name)
        output_file_name = os.path.join(output_path, '{}_cws.json'.format(file_label))
        with codecs.open(output_file_name, mode='w', encoding='utf8') as fw:
            for text in tqdm(texts):
                label, news_content = text.split('\t')
                news_content = news_content.replace(' ', '')
                cws_items = get_baidu_cws(news_content)
                row_json = {
                    'label': label,
                    'cws_items': cws_items,
                    'news_content': news_content
                }
                fw.write('{}\n'.format(json.dumps(row_json, ensure_ascii=False)))
                time.sleep(0.3)
                # break
