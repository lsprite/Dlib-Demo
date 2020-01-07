import json

w = {'name': 'Runoob', 'url': 'www.runoob.com'}
print(f'{w["name"]}: {w["url"]}')

name = 'Runoob'

print('Hello %s' % name)
print('Hello' + ' ' + name)
ss = name.split("n")
print(len(ss))
print(ss[0])
a, b = 0, 1
while b < 10:
    b = b + 1
    print(b, end=",")
b = b + 1
print('end')

for a, b in w.items():
    print(a, b)

for k, v in w.items():
    print(k, v)
print("*" * 10)
print(w)
json_str = json.dumps(w)
print(json_str)
json_data=json.loads(json_str)
print(json_data)