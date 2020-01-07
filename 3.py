# yield解释
# https://blog.csdn.net/mieleizhi0522/article/details/82142856
def foo():
    print('Starting...')
    while True:
        res = yield 4
        print('res:', res)


g = foo()
print(next(g))
print("*" * 20)
print(next(g))
print("*" * 20)
print(next(g))
print("*" * 20)
print(next(g))
print("*" * 20)
print(g.send(10))
print("*" * 20)
print(g.send(10))


def f(a, b, *, c, d):
    return a + b + c


print(f(1, 2, c=3, d=4))

sum = lambda arg1, arg2: arg1 + arg2
print("相加后的值为 : ", sum(10, 20))
