import pandas as pd

from k_choice.simulation import one_choice_simulate_many_runs, two_choice_simulate_many_runs
from k_thinning.environment import run_strategy_multiple_times
from k_thinning.strategies.full_knowledge_DQN_strategy import FullKnowledgeDQNStrategy


def create_comparison(ns=[10, 20, 50, 100], runs=100):  # ms=ns
    df = pd.DataFrame(columns=["strategy"] + list(map(str, ns)))

    one_choice_vals = {str(n): one_choice_simulate_many_runs(runs=runs, n=n, m=n, print_behaviour=False) for n in ns}
    df = df.append({**{"strategy": "one_choice"}, **one_choice_vals}, ignore_index=True)

    two_choice_vals = {str(n): two_choice_simulate_many_runs(runs=runs, n=n, m=n, print_behaviour=False) for n in ns}
    df = df.append({**{"strategy": "two_choice"}, **two_choice_vals}, ignore_index=True)

    two_thinning_vals = {
        str(n): run_strategy_multiple_times(n=n, m=n, k=2, runs=runs, strategy=FullKnowledgeDQNStrategy(n=n, m=n, k=2),
                                       print_behaviour=False) for n in ns}
    df = df.append({**{"strategy": "two_thinning"}, **two_thinning_vals}, ignore_index=True)

    three_thinning_vals = {
        str(n): run_strategy_multiple_times(n=n, m=n, k=3, runs=runs, strategy=FullKnowledgeDQNStrategy(n=n, m=n, k=3),
                                       print_behaviour=False) for n in ns}
    df = df.append({**{"strategy": "three_thinning"}, **three_thinning_vals}, ignore_index=True)

    print(df)
    df.to_csv('Dimitris_comparison.csv', index=False)
    return df


if __name__ == "__main__":
    create_comparison()
