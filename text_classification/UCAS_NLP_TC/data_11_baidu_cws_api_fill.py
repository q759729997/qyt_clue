"""
百度词法分析API，补全未识别出的：
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
        'dev': './data/UCAS_NLP_TC/data_baidu_cws/dev_cws.json',
        'test': './data/UCAS_NLP_TC/data_baidu_cws/test_cws.json',
        'train': './data/UCAS_NLP_TC/data_baidu_cws/train_cws.json',
    }
    for file_label, file_name in train_file_config.items():
        logger.info('开始处理：{}'.format(file_label))
        texts = read_file_texts(file_name)
        with codecs.open(file_name, mode='w', encoding='utf8') as fw:
            for text in tqdm(texts):
                row_json = json.loads(text)
                if len(row_json['cws_items']) == 0:
                    news_content = row_json['news_content']
                    if len(news_content) > 10000:
                        cws_items = get_baidu_cws(news_content[:10000])
                        time.sleep(0.3)
                        cws_items.extend(get_baidu_cws(news_content[10000:]))
                    else:
                        cws_items = get_baidu_cws(news_content)
                    time.sleep(0.3)
                    row_json['cws_items'] = cws_items
                    fw.write('{}\n'.format(json.dumps(row_json, ensure_ascii=False)))
                    time.sleep(0.3)
                else:
                    fw.write('{}\n'.format(text))
