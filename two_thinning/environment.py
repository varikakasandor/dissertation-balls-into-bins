import random
from math import sqrt, log, floor, ceil

from two_thinning.other_strategies.always_accept import AlwaysAcceptStrategy
from two_thinning.other_strategies.the_threshold_strategy import TheThresholdStrategy
from two_thinning.other_strategies.the_relative_threshold_strategy import TheRelativeThresholdStrategy
from two_thinning.other_strategies.drift_strategy import DriftStrategy
from two_thinning.other_strategies.multi_stage_threshold_strategy import MultiStageThresholdStrategy

N = 100
M = 500
LIMIT = sqrt((2 * log(N)) / log(log(N)))  # According to the "The power of thinning in balanced allocation paper"
STRATEGY = AlwaysAcceptStrategy(N, M)
REWARD = max
RUNS = 1000

K = floor(log(log(N)) / (3 * log(log(log(N)))))
T = ceil(M / N)  # TODO: ceil might not be needed
ALPHA = log(T) / (log(log(N)))
ETA = 0
BETA = ALPHA + ETA
EPSILON = (2 * BETA - 1) / (2 * (K + 1))
BETA_K = (2 * BETA - 1 - EPSILON) * K / (2 * K + 1)
L = (log(N)) ** BETA_K
L_0 = 0


def run_strategy(n=N, m=M, strategy=STRATEGY, reward=REWARD):
    loads = [0] * n
    for i in range(m):
        first_choice = random.randrange(n)
        if strategy.decide(first_choice):
            print(f"Bin {first_choice} accepted.")
            final_choice = first_choice
        else:
            print(f"Bin {first_choice} rejected.")
            final_choice = random.randrange(n)

        strategy.note(final_choice)
        loads[final_choice] += 1

    score = reward(loads)
    # print(f"The score of this strategy on this run is {score}")
    return score


def run_strategy_multiple_times(n=N, m=M, runs=RUNS, strategy=STRATEGY, reward=REWARD):
    scores = []
    for _ in range(runs):
        score = run_strategy(n=n, m=m, strategy=strategy, reward=reward)
        scores.append(score)
        strategy.reset()
    avg_score = sum(scores) / runs
    print(f"The average score of this strategy is {avg_score}")


if __name__ == "__main__":
    run_strategy_multiple_times(
        strategy=MultiStageThresholdStrategy(n=N, m=M, T=T, L_0=L_0, l=L))  # I don't understand why it shows yellow,
    # whereas it runs fine
