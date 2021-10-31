from math import log, sqrt


# TODO: find constant factors of the limits from the papers

def k_choice_high_probability_maximum_load(d, n, m):
    if d == 1:
        if m < n / log(n):
            return log(n) / log(n / m)
        elif m < n * log(n):  # TODO: not precise, only for m=n
            return log(n) / log(log(n))
        else:
            return m / n + sqrt(m * log(n) / n)
    else:
        if m <= n:  # TODO: not precise, only for m=n
            return log(log(n)) / log(d)
        else:
            return (m - n) / n + log(log(n)) / log(d)


def one_choice_high_probability_maximum_load(n, m):
    return k_choice_high_probability_maximum_load(1, n, m)


def two_choice_high_probability_maximum_load(n, m):
    return k_choice_high_probability_maximum_load(2, n, m)
