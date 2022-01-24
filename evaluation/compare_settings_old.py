import k_choice
import two_thinning

# TODO: call these evaluation functions


def compare_settings(n=20, m=20, runs=100, reward=max, episodes=100000, epsilon=0.1,alpha = 0.1,version = 'Q'):
    k_choice.simulation.one_choice_simulate_many_runs(n=n,m=m,runs=runs,reward=reward)
    k_choice.simulation.two_choice_simulate_many_runs(n=n,m=m,runs=runs,reward=reward)
    two_thinning.constant_threshold.dp.find_best_constant_threshold(n=n,m=m,reward=reward)
    two_thinning.constant_threshold.RL.evaluate.evaluate(n=n, m=m, episodes=episodes, epsilon=epsilon, runs=runs, reward=reward)
    two_thinning.average_based.RL.evaluate.evaluate(n=n, m=m, episodes=episodes, epsilon=epsilon, runs=runs, reward=reward, alpha=alpha,version=version)
    two_thinning.full_knowledge.dp.find_best_thresholds(n=n, m=m, reward=reward)
    two_thinning.full_knowledge.RL.evaluate.evaluate_best(n=n, m=m, reward=reward, runs=runs)

if __name__=="__main__":
    compare_settings()