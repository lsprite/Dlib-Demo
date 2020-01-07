import keyword

print(keyword.kwlist)
if True:
    print("true")
else:
    print("false")

str = "chi" + \
      "na"
print(str)
print(str[0:2])
print(str[0:])
print(str * 2)
print('hello\nrunoob')  # 使用反斜杠(\)+n转义特殊字符
print(r'hello\nrunoob')
# input("\n\n按下 enter 键后退出。")

import sys

for i in sys.argv:
    print(i)
print('\n python 安装路径为:', sys.path)

from sys import path

print("路径:", path)
print(type(path))
print(isinstance(path, list))

tup = (1,)
print(tup[0])

a = set('abracadabra')
b = set('alacazam')
print(a - b)
print(a | b)
print(a & b)
print(a ^ b)

book={}
book['name']='钢铁'
book['user']='洛夫斯基'
print(book)
print(book['name'])
print(book.keys())
print(book.values())

def a():
    '''sdsadasdsa'''
    return
print(a.__doc__)