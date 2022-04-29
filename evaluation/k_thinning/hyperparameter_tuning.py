import wandb

from k_thinning.full_knowledge.RL.DQN.constants import *
from k_thinning.full_knowledge.RL.DQN.evaluate import evaluate
from k_thinning.full_knowledge.RL.DQN.train import train

N = 5
M = 25
K = 3
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = False

def tuning_function(config=None):
    loss_mapping = {
        "SmoothL1Loss": nn.SmoothL1Loss(),
        "MSELoss": nn.MSELoss(),
        "HuberLoss": nn.HuberLoss(),
        "L1Loss": nn.L1Loss()
    }
    optimizer_mapping = {
        "Adam": torch.optim.Adam,
        "Adagrad": torch.optim.Adagrad,
        "SGD": torch.optim.SGD,
        "RMSprop": torch.optim.RMSprop
    }
    potential_mapping = {
        "max_load": MAX_LOAD_POTENTIAL,
        "std": STD_POTENTIAL,
        "exponential": EXPONENTIAL_POTENTIAL,
        "zero": NO_POTENTIAL
    }
    SAVE_PATH = join((dirname(dirname(abspath(__file__)))), "training_progression",
                     f'{str(datetime.now().strftime("%Y_%m_%d %H_%M_%S_%f"))}_{N}_{M}_{K}')
    with wandb.init(config=config):
        config = wandb.config
        trained_model = train(n=N, m=M, k=K, memory_capacity=config["memory_capacity"],
                              num_episodes=config["train_episodes"],
                              pre_train_episodes=config["pre_train_episodes"],
                              loss_function=loss_mapping[config["loss_function"]], lr=config["lr"],
                              reward_fun=REWARD_FUN, batch_size=config["batch_size"],
                              eps_start=config["eps_start"], pacing_fun=PACING_FUN,
                              eps_end=config["eps_end"], nn_hidden_size=config["hidden_size"], nn_rnn_num_layers=config["num_rnn_layers"],
                              nn_num_lin_layers=config["num_lin_layers"], eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE,
                              eps_decay=config["eps_decay"], optimise_freq=config["optimise_freq"],
                              target_update_freq=config["target_update_freq"], use_normalised=config["use_normalised"],
                              eval_runs=config["eval_runs_train"], patience=config["patience"],
                              potential_fun=potential_mapping[config["potential_fun"]],
                              print_progress=PRINT_PROGRESS, max_threshold=config["max_threshold"],
                              optimizer_method=optimizer_mapping[config["optimizer_method"]],
                              nn_model=NN_MODEL, device=DEVICE, report_wandb=True, save_path=SAVE_PATH)
        score = evaluate(trained_model, n=N, m=M, k=K, reward_fun=REWARD_FUN, eval_runs_eval=config["eval_runs_eval"],
                         use_normalised=config["use_normalised"], max_threshold=config["max_threshold"])
        wandb.log({"score": score})


if __name__ == "__main__":
    wandb.login()

    sweep_config = {
        'method': 'bayes'
    }
    metric = {
        'name': 'score',
        'goal': 'maximize'
    }
    sweep_config['metric'] = metric
    parameters_dict = {
        "train_episodes": {
            "values": [200]
        },
        "patience": {
            "values": [300]
        },
        "eval_runs_eval": {
            "values": [25]  # 30
        },
        "eval_runs_train": {
            "values": [8]
        },
        "pre_train_episodes": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 100
        },
        'batch_size': {
            'distribution': 'int_uniform',
            'min': 16,
            'max': 70
        },
        "eps_start": {
            'distribution': 'uniform',
            'min': 0.05,
            'max': 0.6
        },
        "eps_end": {
            'distribution': 'uniform',
            'min': 0.03,
            'max': 0.1
        },
        "eps_decay": {
            'distribution': 'uniform',
            'min': 1000,  # 100
            'max': 5000
        },
        "target_update_freq": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 30
        },
        "memory_capacity": {
            'distribution': 'int_uniform',
            'min': 300,
            'max': 1000  # 600
        },
        "optimise_freq": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 50  # 30
        },
        "loss_function": {
            "values": ["MSELoss", "HuberLoss", "L1Loss", "SmoothL1Loss"]
        },
        "optimizer_method": {
            "values": ["Adam", "Adagrad", "SGD", "RMSprop"]
        },
        "lr": {
            "distribution": "log_uniform",
            "min": -6,
            "max": -4
        },
        "hidden_size": {
            'distribution': 'int_uniform',
            'min': 32,
            'max': 256
        },
        "num_rnn_layers": {
          'distribution': 'int_uniform',
          'min': 1,
          'max': 3
        },
        "num_lin_layers": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 5
        },
        "potential_fun": {
            "values": ["max_load", "std", "exponential", "zero"]
        },
        "use_normalised": {
            "values": [True, False]
        },
        "max_threshold": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': M
        }
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project=f"k_thinning_{N}_{M}_{K}_final")
    wandb.agent(sweep_id, tuning_function, count=500)
