import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from two_thinning.average_based.RL.invalid_RL.train import train

# ray.init(log_to_driver=False)
ray.init(local_mode=True)

n = 10
m = 20
max_episodes = 3000

reward = max
test_runs = 300
report_frequency = max_episodes // 100


def evaluate_config(config, checkpoint_dir=None):
    train(n=config['n'], m=config['m'], episodes=config['episodes'], epsilon=config['epsilon'], alpha=config['alpha'],
          initial_q_value=config['initial_q_value'], version=config['version'], reward=config['reward'],
          test_runs=config['test_runs'], use_tune=True, report_frequency=config['report_frequency'])
    print(f"{config} done!")


def tune_hyperparameters(n=n, m=m, reward=reward, test_runs=test_runs, max_episodes=max_episodes,
                         report_frequency=report_frequency):
    scheduler = ASHAScheduler(
        metric="avg_test_load",
        mode="min",
        max_t=max_episodes,
        grace_period=1,
        reduction_factor=2)
    config = {
        'n': n,
        'm': m,
        'reward': reward,
        'test_runs': test_runs,
        'report_frequency': report_frequency,
        'episodes': tune.grid_search([max_episodes // 100, max_episodes // 10, max_episodes // 4, max_episodes]),
        'epsilon': tune.grid_search([0, 0.05, 0.1, 0.3]),  # tune.uniform(0, 1),
        'alpha': tune.grid_search([0.2, 0.4, 0.6, 0.8]),  # tune.uniform(0,1),
        'initial_q_value': tune.grid_search([0, m + 1]),
        'version': tune.grid_search(["Q", "Sarsa"])
    }
    result = tune.run(evaluate_config, config=config, metric="avg_test_load", mode="min", num_samples=1,
                      local_dir="individual_runs", verbose=False)  # , scheduler=scheduler)
    (result.results_df).to_csv("./summary.csv")
    print(result.best_config)


if __name__ == '__main__':
    tune_hyperparameters()
