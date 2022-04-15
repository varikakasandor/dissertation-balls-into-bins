from datetime import datetime
from os.path import exists, dirname, join, abspath

import pandas as pd

from evaluation.k_thinning.hyperparameters import get_dqn_hyperparameters, get_threshold_hyperparameters
from k_thinning.environment import run_strategy_multiple_times
from k_thinning.strategies.always_accept_strategy import AlwaysAcceptStrategy
from k_thinning.strategies.dp_strategy import DPStrategy
from k_thinning.strategies.full_knowledge_DQN_strategy import FullKnowledgeDQNStrategy
from k_thinning.strategies.local_reward_optimiser_strategy import LocalRewardOptimiserStrategy
from k_thinning.strategies.mean_thinning_strategy import MeanThinningStrategy
from k_thinning.strategies.random_strategy import RandomStrategy
from k_thinning.strategies.the_threshold_strategy import TheThresholdStrategy

NMKS = ((5, 25, 2), (5, 25, 3), (5, 25, 5), (5, 25, 10), (20, 50, 2), (20, 50, 3), (20, 50, 5), (20, 50, 10))
STRATEGIES = ("always_accept", "random", "local_reward_optimiser", "mean_thinning", "dp", "threshold")  # , "dqn")
RUNS = 100
RE_TRAIN_DQN = 5
PRINT_BEHAVIOUR = False


def REWARD_FUN(loads):
    return -max(loads)


def compare_strategies(nmks=NMKS, runs=RUNS, strategies=STRATEGIES, reward_fun=REWARD_FUN,
                       print_behaviour=PRINT_BEHAVIOUR, re_train_dqn=RE_TRAIN_DQN):
    for n, m, k in nmks:
        for strategy_name in strategies:
            print(f"n={n}, m={m}, k={k} strategy={strategy_name} started.")
            if strategy_name == "dqn":
                if n > 100 or m > 200:  # these are out of the feasible range
                    continue
                hyperparameters = get_dqn_hyperparameters(n=n, m=m, k=k)
                scores = []
                for _ in range(re_train_dqn):
                    save_path = join((dirname(dirname(abspath(__file__)))), "training_progression",
                                     f'{str(datetime.now().strftime("%Y_%m_%d %H_%M_%S_%f"))}_{n}_{m}')
                    strategy = FullKnowledgeDQNStrategy(n=n, m=m, k=k, use_pre_trained=False, save_path=save_path,
                                                        **hyperparameters)
                    curr_scores = run_strategy_multiple_times(n=n, m=m, k=k, runs=runs // re_train_dqn,
                                                              strategy=strategy,
                                                              reward=reward_fun, print_behaviour=print_behaviour)
                    scores.extend(curr_scores)
            else:
                if strategy_name == "always_accept":
                    strategy = AlwaysAcceptStrategy(n=n, m=m, k=k)
                elif strategy_name == "random":
                    strategy = RandomStrategy(n=n, m=m, k=k)
                elif strategy_name == "mean_thinning":
                    strategy = MeanThinningStrategy(n=n, m=m, k=k)
                elif strategy_name == "local_reward_optimiser":
                    strategy = LocalRewardOptimiserStrategy(n=n, m=m, k=k)
                elif strategy_name == "threshold":
                    hyperparameters = get_threshold_hyperparameters(n=n, m=m, k=k)
                    strategy = TheThresholdStrategy(n=n, m=m, k=k, **hyperparameters)
                elif strategy_name == "dp":
                    if n > 50 or m > 70:  # these are out of the feasible range
                        continue
                    strategy = DPStrategy(n=n, m=m, k=k, reward_fun=reward_fun)
                else:
                    raise Exception("No such strategy is known, check spelling!")

                scores = run_strategy_multiple_times(n=n, m=m, k=k, runs=runs, strategy=strategy, reward=reward_fun,
                                                     print_behaviour=print_behaviour)

            df = pd.DataFrame(data=scores, columns=["score"])
            output_path = f'{n}_{m}_{k}_{strategy_name}.csv'
            df.to_csv(output_path, mode='a', index=False, header=not exists(output_path))


if __name__ == "__main__":
    compare_strategies()
