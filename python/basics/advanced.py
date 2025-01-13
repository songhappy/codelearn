import sys
from functools import reduce

x, y = 1, 100
mylyst = [1, 2, 3]
# In a function call:
# * unpacks a list or tuple into position arguments.
# ** unpacks a dictionary into keyword arguments.
lis = [1, 2, 3, 4]
dic = {'a': 10, 'b': 20}
(1, 2, 3, 4)
{'a': 10, 'b': 20}
newlist = [*mylyst, 100, 200]


# Inside a function header:
# * collects all the positional arguments in a tuple.
# ** collects all the keyword arguments in a dictionary.
def functionA(*a, **kw):
    print(a)
    print(kw)


functionA(1, 2, 3, 4, 5, 6, a=2, b=3, c=5)
(1, 2, 3, 4, 5, 6)
{'a': 2, 'c': 5, 'b': 3}
functionA(*lis, **dic)  # it is similar to functionA(1, 2, 3, 4, a=10, b=20)


# define a class inside a function
def build_trainer(name):
    class trainer_cls():
        def __init__(self, config):
            pass

        def _train(self):
            pass

    trainer_cls.__name = name
    trainer_cls.__qualname__ = name
    return trainer_cls


A2cTrainer = build_trainer("a2c")  # works as a class constructor, not an object constructor
trainer = A2cTrainer(config="blabla")


# define a function inside a function
def func1():
    def func2(x):
        return x + 1

    return func2


new_func = func1()
x = (2)
print(x)


# define a function inside a function
class PolicyWithValue():
    pass


def build_policy(env, policy_network, value_work=None):
    def policy_fn(nbath=None, nstes=None):
        policy = PolicyWithValue(env=env)
        pass
        return policy

    return policy_fn()  # eventually return the policy


# inspect
# print(inspect.getsource(Queue))

# magic methods/data model methods and python data model
# magic methods are double underscored methods
from queue import Queue as q


class Queue(q):
    def __repr__(self):
        return f"Queue({self.qsize()})"

    def __add__(self, other):
        self.put(other)


qu = Queue()
print(qu)
qu + 9


# metaclasses
class Foo:
    def show(self):
        print("hi")


def add_atribute(self):
    self.z = 9


# type(name, bases, dict) -> a new type
# how to use type to create a class
Test = type('Test', (Foo,), {"x": 5, "add_attribute": add_atribute})
t = Test()
t.wy = "hello"
print(t.wy)
t.show()
t.add_attribute()
print(t.z)


# meta class can change the class atribute and force in different ways in different user iterface
class Meta(type):
    def __new__(self, class_name, bases, attrs):
        a = {}
        for name, val in attrs.items():
            if name.startswith("__"):
                a[name] = val
            else:
                a[name.upper()] = val  # change atributes to upper class
        return type(class_name, bases, a)


class Dog(metaclass=Meta):
    x = 5
    y = 8

    def hello(self):
        print("hi")


d = Dog()
d.HELLO()


# decorators wrap a function, modifying its behavior
# used when you want to modify behavior of a function without toughing it or changing it, check input and output values, timer
def my_decorator(func):  # pass a function to a function
    def wrapper(*args, **kwargs):  # implement wrapper to change the behavior
        print("Something is happening before the function is called.")
        rv = func(*args, **kwargs)
        print("Something is happening after the function is called.")
        return rv

    return wrapper


# say_whee = my_decorator(say_whee)
@my_decorator
def say_whee():
    print("Whee!")


say_whee()

import time


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        rv = func(*args, **kwargs)
        total = time.time() - start
        print("time:", total)
        return rv

    return wrapper


@timer
def test():
    time.sleep(2)


test()

# generators
x = [i ** 2 for i in range(10)]  # take too much memory


class Gen:  # let you look at one value at a time
    def __init__(self, n):
        self.n = n
        self.last = 0

    def __next__(self):
        return self.next()

    def next(self):
        if self.last == self.n:
            raise StopIteration()
        rv = self.last ** 2
        self.last += 1
        return rv


g = Gen(100)
while True:
    try:
        print(next(g))
    except StopIteration:
        break


# same as
def gen(n):
    for i in range(n):
        yield i ** 2  # pause


g = gen(1000000)
print(next(g))  # next is defined in builtin
print(next(g))
print(sys.getsizeof(g))
my_generator = (i for i in range(10) if i % 2 == 0)
for i in my_generator:
    print(i)

# context managers
# using with, it will close this file at the end no matter break or not
with open("file.txt", "w") as file:
    file.write("hello")

# lambda function
# lambda arguments: expression, usually used only once in the code or as argument for sorted, map, filter and reduce
mult = lambda x, y: x * y
mult(5, 10)
prints2D = [(1, 2), (15, 1), (5, -1), (10, 4)]
sorted = sorted(prints2D, key=lambda x: x[0] + x[1])
# map(func, seq)
a = [1, 2, 3, 4, 5]
b = map(lambda x: x * 2, a)
c = map(lambda x: x % 2 == 0, a)
c = [x for x in a if x % 2 == 0]
# reduce(func, seq)
print(list(b))
d = reduce(lambda x, y: x * y, a)
print(d)

# random numbers
import random

random.seed(1)  # reduceable
a = random.random()
a = random.randint(1, 10)

import secrets  # not reduc["abcdedf"]eble

a = secrets.randbelow(10)
a = secrets.randbits(4)  # largest 15, 4 bits 1111
mylyst = list("abcdedf")
print(mylyst)
a = secrets.choice(mylyst)
print(a)

import numpy as np

np.random.seed(1)  # reproduceable
a = np.random.rand(3)
print(a)
a = np.random.randint(0, 10, (3, 4))
print(a)
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr)
np.random.shuffle(arr)
print(arr)

## format
print(f'{x:02} {x * x:3} {x * x * x:4}')
# 09  81  729
time1 = time.time()
time2 = time.time() + 10
print(f"perf training time: {(time2 - time1):.2f}")
