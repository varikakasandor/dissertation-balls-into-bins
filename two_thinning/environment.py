import random
from math import sqrt, log

from two_thinning.other_strategies.always_accept import AlwaysAcceptStrategy
from two_thinning.other_strategies.the_threshold_strategy import TheThresholdStrategy
from two_thinning.other_strategies.the_relative_threshold_strategy import TheRelativeThresholdStrategy

N = 10
M = 20
LIMIT = sqrt((2*log(N))/log(log(N)))  # According to the "The power of thinning in balanced allocation paper"
STRATEGY = AlwaysAcceptStrategy(N,M)
REWARD = max
RUNS = 100


def run_strategy(n=N, m=M, strategy=STRATEGY, reward=REWARD):
    loads = [0] * n
    for i in range(m):
        first_choice = random.randrange(n)
        if (strategy.decide(first_choice)):
            loads[first_choice] += 1
        else:
            second_choice = random.randrange(n)
            loads[second_choice] += 1

    score = reward(loads)
    # print(f"The score of this strategy on this run is {score}")
    return score


def run_strategy_multiple_times(n=N, m=M, runs=RUNS, strategy=STRATEGY, reward=REWARD):
    scores = [run_strategy(n=n, m=m, strategy=strategy, reward=reward) for _ in range(runs)]
    avg_score = sum(scores) / runs
    print(f"The average score of this strategy is {avg_score}")


if __name__=="__main__":
    run_strategy_multiple_times(strategy=TheRelativeThresholdStrategy(n=N,m=M,limit=LIMIT)) # I don't understand why it shows yellow, whereas it runs fine
