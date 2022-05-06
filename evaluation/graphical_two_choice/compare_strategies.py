from datetime import datetime
from os.path import exists, dirname, join, abspath

import pandas as pd

from evaluation.graphical_two_choice.hyperparameters import get_dqn_hyperparameters
from k_choice.graphical.two_choice.environment import run_strategy_multiple_times
from k_choice.graphical.two_choice.strategies.greedy_strategy import GreedyStrategy
from k_choice.graphical.two_choice.strategies.dp_strategy import DPStrategy
from k_choice.graphical.two_choice.strategies.full_knowledge_DQN_strategy import FullKnowledgeDQNStrategy
from k_choice.graphical.two_choice.strategies.local_reward_optimiser_strategy import LocalRewardOptimiserStrategy
from k_choice.graphical.two_choice.strategies.random_strategy import RandomStrategy

from k_choice.graphical.two_choice.graphs.complete_graph import CompleteGraph
from k_choice.graphical.two_choice.graphs.random_regular_graph import RandomRegularGraph  # TODO: analyse separately
from k_choice.graphical.two_choice.graphs.cycle import Cycle
from k_choice.graphical.two_choice.graphs.hypercube import HyperCube

GMS = ((Cycle(16), 50), (HyperCube(16), 50), (CompleteGraph(16), 50))
# ,(Cycle(32), 32), (HyperCube(32), 32), (CompleteGraph(32), 32))
#, (Cycle(4), 25), (HyperCube(4), 25), (CompleteGraph(4), 25))

STRATEGIES = ("dqn", )  #("greedy", "random", "local_reward_optimiser", "dp", "dqn")
RUNS = 500
RE_TRAIN_DQN = 1
PRINT_BEHAVIOUR = False


def REWARD_FUN(loads):
    return -max(loads)


def compare_strategies(gms=GMS, runs=RUNS, strategies=STRATEGIES, reward_fun=REWARD_FUN,
                       print_behaviour=PRINT_BEHAVIOUR, re_train_dqn=RE_TRAIN_DQN):
    for graph, m in gms:
        for strategy_name in strategies:
            print(f"graph={graph}, m={m}, strategy={strategy_name} started.")
            if strategy_name == "dqn":
                hyperparameters = get_dqn_hyperparameters(graph=graph, m=m)
                scores = []
                for _ in range(re_train_dqn):
                    strategy = FullKnowledgeDQNStrategy(graph=graph, m=m, use_pre_trained=False, **hyperparameters)
                    curr_scores = run_strategy_multiple_times(graph=graph, m=m, runs=runs // re_train_dqn,
                                                              strategy=strategy,
                                                              reward_fun=reward_fun, print_behaviour=print_behaviour)
                    scores.extend(curr_scores)
            else:
                if strategy_name == "greedy":
                    strategy = GreedyStrategy(graph=graph, m=m)
                elif strategy_name == "random":
                    strategy = RandomStrategy(graph=graph, m=m)
                elif strategy_name == "local_reward_optimiser":
                    strategy = LocalRewardOptimiserStrategy(graph=graph, m=m)
                elif strategy_name == "dp":
                    if graph.n >= 8:
                        continue
                    strategy = DPStrategy(graph=graph, m=m, reward_fun=reward_fun)
                else:
                    raise Exception("No such strategy is known, check spelling!")

                scores = run_strategy_multiple_times(graph=graph, m=m, runs=runs, strategy=strategy, reward_fun=reward_fun,
                                                     print_behaviour=print_behaviour)

            df = pd.DataFrame(data=scores, columns=["score"])
            output_path = f'data/{graph.name}_{graph.n}_{m}_{strategy_name}.csv'
            df.to_csv(output_path, mode='a', index=False, header=not exists(output_path))


if __name__ == "__main__":
    compare_strategies()
