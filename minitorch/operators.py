"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x, y) -> float:
    return x * y


def id(x) -> float:
    return x


def add(x, y) -> float:
    return x + y


def neg(x) -> float:
    return -float(x)


def lt(x, y) -> float:
    if x < y:
        return 1.0
    return 0.0


def eq(x, y) -> float:
    if x == y:
        return 1.0
    return 0.0


def max(x, y) -> float:
    if x > y:
        return x
    return y


def is_close(x, y):
    return abs(x - y) < 1e-2


def sigmoid(x) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def relu(x) -> float:
    if x > 0:
        return x
    return 0.0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x, d) -> float:
    return d / x


def inv(x) -> float:
    return 1 / x


def inv_back(x, d) -> float:
    return -d / x ** 2


def relu_back(x, d) -> float:
    if x > 0:
        return d
    return 0.0


def sigmoid_back(x, d) -> float:
    return d * exp(-x) / ((1 + exp(-x)) ** 2)


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(func):
    return lambda iterable: [func(element) for element in iterable]


def zipWith(func):
    return lambda array1, array2: (func(element_from_array1, element_from_array2) for element_from_array1, element_from_array2 in zip(array1, array2))


def reduce_implementation(func, init_value, values):
    for value in values:
        init_value = func(init_value, value)
    return init_value


def reduce(func, init_value):
    return lambda values: reduce_implementation(func, init_value, values)


def negList(array):
    return map(neg)(array)


def addLists(list1, list2):
    return zipWith(add)(list1, list2)


def sum(array):
    return reduce(add, 0)(array)


def prod(array):
    return reduce(mul, 1)(array)
