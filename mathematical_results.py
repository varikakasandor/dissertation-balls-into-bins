import math

def one_choice_maths(n):
    return math.log(n)/(math.log(math.log(n)))


def two_thinning_constant_threshold_maths(n):
    return math.sqrt(2*math.log(n)/(math.log(math.log(n))))