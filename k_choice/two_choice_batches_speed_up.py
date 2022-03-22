from math import sqrt

N = 100
M = 500
reward = max


def create_alias_table(probs):
    return []


def simulate_one_run(n=N, m=M):
    sqrt_n = sqrt(n)
    loads = [0] * n
    batches = m // sqrt_n
    left_over = m - batches * sqrt_n
