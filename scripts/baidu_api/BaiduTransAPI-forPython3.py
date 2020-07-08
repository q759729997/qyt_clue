#百度通用翻译API,不包含词典、tts语音合成等资源，如有相关需求请联系translate_api@baidu.com
# coding=utf-8

import http.client
import hashlib
import urllib
import random
import json
import sys

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级
from data import baidu_config  # noqa

appid = baidu_config.trans_APP_ID  # 填写你的appid
secretKey = baidu_config.trans_SECRET_KEY  # 填写你的密钥

httpClient = None
myurl = '/api/trans/vip/translate'

fromLang = 'auto'   # 原文语种
toLang = 'en'   # 译文语种
salt = random.randint(32768, 65536)
news_content = '这 是 李咏 在 微博 发 的 最后 一 张 照片 。20 年 前 的 冬天 ， 央视 一 档 益智类 节目 《 幸运 52 》 开播 ， 留 着 一头 卷发 的 主持人 李咏 ， 一边 挥舞 着 手 里 的 小锤 ， 一边 笑 着 询问 嘉宾 ， “ 是 砸 金蛋 还是 银蛋 呢 ？ ” 这 句 询问 ， 包裹 着 紧张 和 亲切 的 情感 ， 留在 了 一代 人 的 记忆 里 。20 年 后 的 10月 25日 ， 李咏因患 癌症 在 美国 去世 ， 年 仅 50 岁 。'
q = news_content.replace(' ', '')
# 单次翻译文本长度限定为6000字节以内（汉字约为2000个字符）
sign = appid + q + str(salt) + secretKey
sign = hashlib.md5(sign.encode()).hexdigest()
myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
salt) + '&sign=' + sign

try:
    httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
    httpClient.request('GET', myurl)

    # response是HTTPResponse对象
    response = httpClient.getresponse()
    result_all = response.read().decode("utf-8")
    result = json.loads(result_all)
    print(result)
except Exception as e:
    print(e)
finally:
    if httpClient:
        httpClient.close()
