import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from two_thinning.average_based.RL.train import train

#ray.init(log_to_driver=False)
ray.init(local_mode=True)

n = 10
m = 20
episodes = 3000

reward = max
test_runs=300

def evaluate_config(config, checkpoint_dir=None):
    train(n=config['n'], m=config['m'], episodes=config['episodes'], epsilon=config['epsilon'], alpha=config['alpha'],
          version=config['version'], reward=config['reward'], test_runs=config['test_runs'], use_tune=True)


def tune_hyperparameters(n=n, m=m, reward=reward, test_runs=test_runs, max_episodes=episodes):
    scheduler = ASHAScheduler(
        metric="avg_test_load",
        mode="min",
        max_t=max_episodes,
        grace_period=1,
        reduction_factor=2)
    config = {
        'n': n,
        'm': m,
        'episodes': max_episodes,
        'reward': reward,
        'test_runs': test_runs,
        'epsilon': tune.uniform(0, 1),
        'alpha': tune.uniform(0,1),
        'version': tune.grid_search(["Q", "Sarsa"]),
    }
    result = tune.run(evaluate_config, config=config, metric="avg_test_load", mode="min", num_samples=32,
                      local_dir="./individual_runs", verbose=False)#, scheduler=scheduler)
    (result.results_df).to_csv("./summary.csv")
    print(result.best_config)

if __name__=='__main__':
    tune_hyperparameters()