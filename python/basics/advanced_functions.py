# 1. return a function
def func1():
    def func2(x):
        return x + 1

    return func2


new_func = func1()
func1()(1)
new_func(1)


def func1(x):
    def func2(x1):
        return x

    return func2


print("--------")
print(func1(10)(20))


def func3(x):
    return x


def builder(name):
    if callable(name):
        return name
    else:
        print("wrong")


# print(builder(func3))

def sf01(a):
    return a + 1


x = 1
y = 2
z = 3
out = [*map(sf01, (x, y, z))]
print(out)
x = []
x.append([1, 2, 3])
x.append([2, 3, 4])
import numpy as np

y = np.concatenate(x)
print(y)


def myFun1(**kwargs):
    print(kwargs)
    for key, value in kwargs.items():
        print("%s == %s" % (key, value))


myFun1(first='Geeks', mid='for', last='Geeks')


def myFun(arg1, arg2, arg3):
    print("arg1:", arg1)
    print("arg2:", arg2)
    print("arg3:", arg3)


def myFun2(*arg):
    print(arg)


# Now we can use *args or **kwargs to
# pass arguments to this function :
args = ("Geeks", "for", "Geeks")
myFun(*args)
kwargs = {"arg1": 2, "arg2": 3, "arg3": 4}
myFun1(**kwargs)

test_keys = ["Rash", "Kil", "Varsha"]
test_values = [1, 4, 5]
res = dict(zip(test_keys, test_values))
myFun2(test_keys)

print(res)
