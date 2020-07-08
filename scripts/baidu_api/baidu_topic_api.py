"""
百度文本分类API：
pip install baidu-aip
"""
import time
import sys
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


def get_baidu_topic(text):
    for i in range(20):
        try:
            text = text.encode('gbk', errors='ignore').decode('gbk', errors='ignore')  # 去掉GBK不识别的字符串，该接口接收GBK格式
            cws_result = client.topic('凤凰新闻', text)
            if 'item' in cws_result:
                return cws_result['item']
            else:
                print(cws_result)
                continue
        except Exception as e:
            time.sleep(0.5)
            print('text:{}, i:{}, exception：{}'.format(text, i, e))
            traceback.print_exc()
    return {}


if __name__ == "__main__":
    news_content = '这 是 李咏 在 微博 发 的 最后 一 张 照片 。20 年 前 的 冬天 ， 央视 一 档 益智类 节目 《 幸运 52 》 开播 ， 留 着 一头 卷发 的 主持人 李咏 ， 一边 挥舞 着 手 里 的 小锤 ， 一边 笑 着 询问 嘉宾 ， “ 是 砸 金蛋 还是 银蛋 呢 ？ ” 这 句 询问 ， 包裹 着 紧张 和 亲切 的 情感 ， 留在 了 一代 人 的 记忆 里 。20 年 后 的 10月 25日 ， 李咏因患 癌症 在 美国 去世 ， 年 仅 50 岁 。'
    topic_items = get_baidu_topic(news_content.replace(' ', ''))
    print(topic_items)
