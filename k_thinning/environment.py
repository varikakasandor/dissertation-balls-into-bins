import random

from k_thinning.strategies.always_accept_strategy import AlwaysAcceptStrategy
from k_thinning.strategies.full_knowledge_DQN_strategy import FullKnowledgeDQNStrategy
from k_thinning.strategies.random_strategy import RandomStrategy
from k_thinning.strategies.mean_thinning_strategy import MeanThinningStrategy
from k_thinning.strategies.the_threshold_strategy import TheThresholdStrategy
from k_thinning.strategies.dp_strategy import DPStrategy

N = 10
M = 30
K = 3
STRATEGY = AlwaysAcceptStrategy(N, M, K)
RUNS = 5
PRINT_BEHAVIOUR = True


def REWARD_FUN(loads):
    return -max(loads)


def run_strategy(n=N, m=M, k=K, strategy=STRATEGY, reward=REWARD_FUN, print_behaviour=PRINT_BEHAVIOUR):
    loads = [0] * n
    for i in range(m):
        choices_left = k
        final_choice = None
        while choices_left > 1:
            final_choice = random.randrange(n)
            if print_behaviour:
                print(f"Ball number {i}, choice {k - choices_left + 1}, load {loads[final_choice]}", end=": ")
            if strategy.decide_(final_choice):
                if print_behaviour:
                    print(f"ACCEPTED")
                break
            else:
                if print_behaviour:
                    print(f"REJECTED")
                choices_left -= 1

        if choices_left == 1:
            final_choice = random.randrange(n)

        strategy.note_(final_choice)
        loads[final_choice] += 1

    score = reward(loads)
    # print(f"The score of this strategy on this run is {score}")
    return score


def run_strategy_multiple_times(n=N, m=M, k=K, runs=RUNS, strategy=STRATEGY, reward=REWARD_FUN,
                                print_behaviour=PRINT_BEHAVIOUR):
    scores = []
    for i in range(runs):
        score = run_strategy(n=n, m=m, k=k, strategy=strategy, reward=reward, print_behaviour=print_behaviour)
        scores.append(score)
        strategy.reset_()
    avg_score = sum(scores) / runs
    if print_behaviour:
        print(f"The average score of this strategy is {avg_score}")
        print(f"The average normalised max load of this strategy is {-avg_score - m / n}.")
    return scores


if __name__ == "__main__":
    run_strategy_multiple_times(
        strategy=DPStrategy(n=N, m=M, k=K))  # I don't understand why it shows yellow,
    # whereas it runs fine
