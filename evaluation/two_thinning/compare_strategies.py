from os.path import exists, dirname, join, abspath
from datetime import datetime
import pandas as pd

from two_thinning.environment import run_strategy_multiple_times

from two_thinning.strategies.always_accept_strategy import AlwaysAcceptStrategy
from two_thinning.strategies.local_reward_optimiser_strategy import LocalRewardOptimiserStrategy
from two_thinning.strategies.full_knowledge_DQN_strategy import FullKnowledgeDQNStrategy
from two_thinning.strategies.dp_strategy import DPStrategy
from two_thinning.strategies.random_strategy import RandomStrategy
from two_thinning.strategies.the_threshold_strategy import TheThresholdStrategy
from two_thinning.strategies.mean_thinning_strategy import MeanThinningStrategy

from two_thinning.full_knowledge.RL.DQN.constants import MAX_LOAD_POTENTIAL

from evaluation.two_thinning.hyperparameters import get_dqn_hyperparameters, get_threshold_hyperparameters

NMS = ((50, 2500), )  # ((5, 5), (5, 10), (5, 25), (20, 20), (20, 60), (20, 400), (50, 50), (50, 200), (50, 2500))
STRATEGIES = ("dqn", )  # ("always_accept", "random", "local_reward_optimiser", "mean_thinning", "dqn", "threshold")
RUNS = 100
RE_TRAIN_DQN = 1
PRINT_BEHAVIOUR = False


def REWARD_FUN(loads):
    return -max(loads)


def compare_strategies(nms=NMS, runs=RUNS, strategies=STRATEGIES, reward_fun=REWARD_FUN,
                       print_behaviour=PRINT_BEHAVIOUR, re_train_dqn=RE_TRAIN_DQN):
    for n, m in nms:
        for strategy_name in strategies:
            print(f"n={n}, m={m}, strategy={strategy_name} started.")
            if strategy_name == "dqn":
                hyperparameters = get_dqn_hyperparameters(n=n, m=m)
                scores = []
                for _ in range(re_train_dqn):
                    save_path = join((dirname(dirname(abspath(__file__)))), "training_progression",
                                     f'{str(datetime.now().strftime("%Y_%m_%d %H_%M_%S_%f"))}_{n}_{m}')
                    strategy = FullKnowledgeDQNStrategy(n=n, m=m, use_pre_trained=False, save_path=save_path,
                                                        **hyperparameters)
                    curr_scores = run_strategy_multiple_times(n=n, m=m, runs=runs // re_train_dqn, strategy=strategy,
                                                              reward=reward_fun,
                                                              print_behaviour=print_behaviour)
                    scores.extend(curr_scores)
            else:
                if strategy_name == "always_accept":
                    strategy = AlwaysAcceptStrategy(n=n, m=m)
                elif strategy_name == "random":
                    strategy = RandomStrategy(n=n, m=m)
                elif strategy_name == "mean_thinning":
                    strategy = MeanThinningStrategy(n=n, m=m)
                elif strategy_name == "local_reward_optimiser":
                    strategy = LocalRewardOptimiserStrategy(n=n, m=m, reward_fun=reward_fun,
                                                            potential_fun=MAX_LOAD_POTENTIAL)
                elif strategy_name == "threshold":
                    hyperparameters = get_threshold_hyperparameters(n=n, m=m)
                    strategy = TheThresholdStrategy(n=n, m=m, reward_fun=reward_fun, **hyperparameters)
                elif strategy_name == "dp":
                    if n > 50 or m > 70:  # these are out of the feasible range
                        continue
                    strategy = DPStrategy(n=n, m=m, reward_fun=reward_fun)
                else:
                    raise Exception("No such strategy is known, check spelling!")

                scores = run_strategy_multiple_times(n=n, m=m, runs=runs, strategy=strategy, reward=reward_fun,
                                                     print_behaviour=print_behaviour)

            df = pd.DataFrame(data=scores, columns=["score"])
            output_path = f'data/{n}_{m}_{strategy_name}.csv'
            df.to_csv(output_path, mode='a', index=False, header=not exists(output_path))


if __name__ == "__main__":
    compare_strategies()
