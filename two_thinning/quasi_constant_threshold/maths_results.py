import math


# TODO: find constant factors of the limits from the papers

def optimal_threshold(n):
    return math.sqrt(2 * math.log(n) / (math.log(math.log(n))))

def optimal_result(n):
    return math.sqrt(8 * math.log(n) / (math.log(math.log(n))))
