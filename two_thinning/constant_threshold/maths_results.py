import math


def optimal_threshold(n):
    return math.sqrt(2*math.log(n)/(math.log(math.log(n))))