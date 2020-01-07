import sys

a = [66.25, 333, 333, 1, 1234.5]
print(a.count(333), a.count(66.25), a.count(0))
a.insert(2, -1)
print(a)
a.append(1)
print(a)
print(a.index(333))
a.reverse()
print(a)
a.sort()
print(a)
print('pop:', a.pop(3))
print(a)
del a[2:5]
print(a)
print(dir())

try:
    0 / 0
except:
    print("Unexpected error:", sys.exc_info())
