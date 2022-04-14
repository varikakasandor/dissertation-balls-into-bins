import random
from os import mkdir
import numpy as np

from two_thinning.full_knowledge.RL.DQN.constants import *
from two_thinning.strategies.always_accept_strategy import AlwaysAcceptStrategy
from two_thinning.strategies.local_reward_optimiser_strategy import LocalRewardOptimiserStrategy
from two_thinning.strategies.full_knowledge_DQN_strategy import FullKnowledgeDQNStrategy
from two_thinning.strategies.dp_strategy import DPStrategy
from two_thinning.strategies.random_strategy import RandomStrategy

N = 5
M = 10
STRATEGY = AlwaysAcceptStrategy(N, M)
RUNS = 30
PRINT_BEHAVIOUR = False

def REWARD_FUN(loads):
    return -max(loads)


def run_strategy(time_stamp, run_id, n=N, m=M, strategy=STRATEGY, reward=REWARD_FUN, print_behaviour=PRINT_BEHAVIOUR):
    loads = [0] * n
    for i in range(m):
        first_choice = random.randrange(n)
        if print_behaviour:
            print(f"Ball number {i}, first choice load {loads[first_choice]}", end=": ")
        if strategy.decide_(first_choice):
            if print_behaviour:
                print(f"ACCEPTED")
            final_choice = first_choice
        else:
            if print_behaviour:
                print(f"REJECTED")
            final_choice = random.randrange(n)

        strategy.note_(final_choice)
        loads[final_choice] += 1

    score = reward(loads)
    save_path = join(dirname(dirname(abspath(__file__))), "evaluation", "analyses", time_stamp, run_id)
    strategy.create_analyses_(save_path=save_path)
    return score


def run_strategy_multiple_times(n=N, m=M, runs=RUNS, strategy=STRATEGY, reward=REWARD_FUN, print_behaviour=PRINT_BEHAVIOUR):
    time_stamp = str(datetime.now().strftime("%Y_%m_%d %H_%M_%S_%f"))
    mkdir(join(dirname(dirname(abspath(__file__))), "evaluation", "analyses", time_stamp))
    scores = []
    for i in range(runs):
        score = run_strategy(n=n, m=m, strategy=strategy, reward=reward, print_behaviour=print_behaviour,
                             time_stamp=time_stamp, run_id=str(i))
        scores.append(score)
        strategy.reset_()
    avg_score = sum(scores) / runs
    save_path = join(dirname(dirname(abspath(__file__))), "evaluation", "analyses", time_stamp, "summary")
    strategy.create_summary_(save_path)
    if print_behaviour:
        print(f"The average score of this strategy is {avg_score}")
        print(f"The average normalised max load of this strategy is {-avg_score - m / n}.")
    return scores


if __name__ == "__main__":
    run_strategy_multiple_times(strategy=RandomStrategy(n=N, m=M))  # I don't understand why it shows
    # yellow, whereas it runs fine
    # scores = {alpha: run_strategy_multiple_times(
    # strategy=LocalRewardOptimiserStrategy(n=N, m=M, potential_fun=lambda x: EXPONENTIAL_POTENTIAL(x, alpha=alpha)),
    # print_behaviour=False) for alpha in np.linspace(0, 2, 10)}
