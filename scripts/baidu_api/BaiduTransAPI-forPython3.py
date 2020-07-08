#百度通用翻译API,不包含词典、tts语音合成等资源，如有相关需求请联系translate_api@baidu.com
# coding=utf-8

import http.client
import hashlib
import urllib
import random
import json
import sys
import traceback

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级
from data import baidu_config  # noqa

appid = baidu_config.trans_APP_ID  # 填写你的appid
secretKey = baidu_config.trans_SECRET_KEY  # 填写你的密钥

httpClient = None
myurl = '/api/trans/vip/translate'

fromLang = 'auto'   # 原文语种 auto
toLang = 'en'   # 译文语种
salt = random.randint(32768, 65536)
news_content = '温情励志影片《灵魂的救赎》即将于1月11日暖心上映，影片由张珈铭担任总制片人，知名导演杨真执导，王迅、黄小蕾、以及童星张峻豪主演。近日，影片最新发布了一支制作特辑，独家揭秘了影片的幕后故事，王迅、黄小蕾畅聊表演心得与感触。此次王迅在影片中所饰演的角色何国典是一个典型的四川农村中年男人，他憨厚朴实、一心为家，给予了家庭最无私的爱。王迅在制作特辑中提到自己饰演的角色，“他没有很辉煌的事业，也没有很高的收入，但是他有一颗真正爱这个家庭的心”。何国典就像每一个平凡着却又用爱感动人的万千父亲一样，真实地生活在我们的身边。而人物经历与情感的真实也深深打动了王迅，这也是他参演的重要原因之一。在制作特辑中，王迅被众人一路追赶，还被打倒在地，痛苦的接受众人的拳打脚踢，表现得十分敬业。导演杨真评价王迅，“是一个特别有创造性的演员，用写实的表演震撼人心。”王迅饰演的角色何国典还有一段特殊经历，他是汶川地震的受害者，在地震中他失去了独生子。而王迅本人也是2008年这场灾难的亲历者，所以他更能体会到人物的真情实感。杨真导演介绍到，“地震时，王迅就在事发地，他见证了抗震救灾的现场。”为了真实还原抗震救灾现场，剧组在拍摄这场戏的当天，派来了两辆大型洒水车，而演员王迅需要在大雨滂沱中找寻失踪的孩子，拍摄时为了达到真实效果并没有增添保护措施，在寒冷的天气中，王迅被滂沱大雨淋成了“落汤鸡”，浑身上下都被水浸透了，被冻得瑟瑟发抖。'
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
    print('result_all:{}'.format(result_all))
    result = json.loads(result_all)
    print(result)
except Exception as e:
    print(e)
    traceback.print_exc()
finally:
    if httpClient:
        httpClient.close()
