# from urllib.request import urlopen
#
# response = urlopen("http://www.baidu.com")
# print(type(response))
# print(response.status)
# print(response.getheaders())
# print(response.getheader('Server'))
# html = response.read()
# print(html)

import requests

url = 'http://item.jd.com/2967929.html'
try:
    r = requests.get(url, timeout=30) # 请求超时时间为30秒
    r.raise_for_status() # 如果状态不是200，则引发异常
    r.encoding = r.apparent_encoding# 配置编码
    print(r.text[:1000])  # 部分信息
except Exception as e:
    print(e)
