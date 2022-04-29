import random

from k_choice.graphical.two_choice.strategies.greedy_strategy import GreedyStrategy
from k_choice.graphical.two_choice.strategies.full_knowledge_DQN_strategy import FullKnowledgeDQNStrategy
from k_choice.graphical.two_choice.strategies.dp_strategy import DPStrategy

from k_choice.graphical.two_choice.graphs.cycle import Cycle
from k_choice.graphical.two_choice.graphs.hypercube import HyperCube
from k_choice.graphical.two_choice.graphs.random_regular_graph import RandomRegularGraph

M = 32
N = 32
D = 2
GRAPH = RandomRegularGraph(n=N, d=D)
STRATEGY = GreedyStrategy(GRAPH, M)  # , use_pre_trained=False) Greedy(GRAPH, M) #
RUNS = 1000
PRINT_BEHAVIOUR = True

def REWARD_FUN(x):
    return -max(x)


def run_strategy(graph=GRAPH, m=M, strategy=STRATEGY, reward_fun=REWARD_FUN, print_behaviour=PRINT_BEHAVIOUR):
    loads = [0] * graph.n
    random_edges = random.choices(graph.edge_list, k=m)
    for i in range(m):
        bin1, bin2 = random_edges[i]
        if print_behaviour:
            print(f"Ball number {i}, bin1 load {loads[bin1]}, bin2 load {loads[bin2]}, chosen", end=": ")
        if strategy.decide_(bin1, bin2):
            chosen_bin = bin1
            if print_behaviour:
                print(f"FIRST")
        else:
            chosen_bin = bin2
            if print_behaviour:
                print(f"SECOND")

        loads[chosen_bin] += 1

    score = reward_fun(loads)
    return score


def run_strategy_multiple_times(graph=GRAPH, m=M, runs=RUNS, strategy=STRATEGY, reward_fun=REWARD_FUN,
                                print_behaviour=PRINT_BEHAVIOUR):
    scores = []
    for _ in range(runs):
        score = run_strategy(graph=graph, m=m, strategy=strategy, reward_fun=reward_fun, print_behaviour=print_behaviour)
        scores.append(score)
        strategy.reset_()
    avg_score = sum(scores) / runs
    if print_behaviour:
        print(f"The average score of this strategy is {avg_score}")
        print(f"The average normalised max load of this strategy is {-avg_score - m / graph.n}.")
    return scores


if __name__ == "__main__":
    run_strategy_multiple_times()
