import pandas as pd

from k_choice.simulation import one_choice_simulate_many_runs, two_choice_simulate_many_runs
from k_thinning.environment import run_strategy_multiple_times as run_strategy_multiple_times_k
from two_thinning.environment import run_strategy_multiple_times as run_strategy_multiple_times_2
from k_thinning.strategies.full_knowledge_DQN_strategy import FullKnowledgeDQNStrategy
from two_thinning.strategies.full_knowledge_rare_change_DQN_strategy import FullKnowledgeRareChangeDQNStrategy
from two_thinning.strategies.the_threshold_strategy import TheThresholdStrategy


def create_comparison(save_name, ns=[10, 20, 50, 100], ms=[10, 20, 50, 100], runs=10):
    df = pd.DataFrame(columns=["strategy"] + list(map(str, ns)))

    one_choice_vals = {str(n): one_choice_simulate_many_runs(runs=runs, n=n, m=m, print_behaviour=False) for n,m in zip(ns, ms)}
    df = df.append({**{"strategy": "one_choice"}, **one_choice_vals}, ignore_index=True)

    two_choice_vals = {str(n): two_choice_simulate_many_runs(runs=runs, n=n, m=m, print_behaviour=False) for n,m in zip(ns, ms)}
    df = df.append({**{"strategy": "two_choice"}, **two_choice_vals}, ignore_index=True)

    for limit in range(0, 4):
        two_thinning_fixed_vals = {
            str(n): run_strategy_multiple_times_2(n=n, m=m, runs=runs,
                                                  strategy=TheThresholdStrategy(n=n, m=m, limit=limit),
                                                  print_behaviour=False) for n,m in zip(ns, ms)}
        df = df.append({**{"strategy": f"two_thinning_fixed_limit_{limit}"}, **two_thinning_fixed_vals},
                       ignore_index=True)

    two_thinning_vals = {
        str(n): run_strategy_multiple_times_k(n=n, m=m, k=2, runs=runs,
                                              strategy=FullKnowledgeDQNStrategy(n=n, m=m, k=2),
                                              print_behaviour=False) for n,m in zip(ns, ms)}
    df = df.append({**{"strategy": "two_thinning"}, **two_thinning_vals}, ignore_index=True)

    three_thinning_vals = {
        str(n): run_strategy_multiple_times_k(n=n, m=m, k=3, runs=runs,
                                              strategy=FullKnowledgeDQNStrategy(n=n, m=m, k=3),
                                              print_behaviour=False) for n,m in zip(ns, ms)}
    df = df.append({**{"strategy": "three_thinning"}, **three_thinning_vals}, ignore_index=True)

    two_thinning_rare_change_vals = {
        str(n): run_strategy_multiple_times_2(n=n, m=m, runs=runs,
                                              strategy=FullKnowledgeRareChangeDQNStrategy(n=n, m=m),
                                              print_behaviour=False) for n,m in zip(ns, ms)}
    df = df.append({**{"strategy": "two_thinning_rare_change"}, **two_thinning_rare_change_vals}, ignore_index=True)

    print(df)
    df.to_csv(f'{save_name}.csv', index=False)
    return df


if __name__ == "__main__":
    create_comparison("comparison_m=n^2", ns=[3, 5, 8, 10, 12], ms=[9, 25, 64, 100, 144])
