import torch
import numpy as np
import os

from two_thinning.full_knowledge.RL.neural_network import TwoThinningNet
from two_thinning.full_knowledge.RL.train import train

n = 10
m = n
epsilon = 0.1  # TODO: set (exponential) decay


def reward(x):
    return -np.max(x)


train_episodes = 3000
runs = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_best_model_path(n=n, m=m):
    best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models", f"best_{n}_{m}.pth")
    return best_model_path

def evaluate(model, n=n, m=m, reward=reward, runs=runs):
    max_loads = []
    for _ in range(runs):
        loads = np.zeros(n)
        for i in range(m):
            options = model(torch.from_numpy(loads).double())
            # print(f"The options are: {options}")
            a = torch.argmax(options)
            # print(f"With load vector {loads}, the model chooses a threshold {a}")
            randomly_selected = np.random.randint(n)
            if loads[randomly_selected] <= a:
                loads[randomly_selected] += 1
            else:
                loads[np.random.randint(n)] += 1
        max_loads.append(-reward(loads))

    avg_max_load = sum(max_loads) / len(max_loads)
    return avg_max_load


def load_best_model(n=n, m=m, device=device):
    best_model = TwoThinningNet(n, m)
    best_model.to(device).double()
    best_model.load_state_dict(torch.load(get_best_model_path(n=n, m=m)))
    best_model.eval()
    return best_model


def evaluate_best(n=n, m=m, reward=reward, device=device, runs=runs):
    avg_load = -evaluate(load_best_model(n=n, m=m, device=device), n=n, m=m, reward=reward, runs=runs)
    print(f"With {m} balls and {n} bins the average maximum load of the derived greedy full knowledge policy is {avg_load}")
    return avg_load


def compare(n=n, m=m, epsilon=epsilon, reward=reward, train_episodes=train_episodes, device=device):

    current_model = train(n=n, m=m, epsilon=epsilon, reward=reward, episodes=train_episodes, device=device)
    current_model_performance = evaluate(current_model, n=n, m=m, reward=reward, runs=runs)
    print(f"The average maximum load of the current model is {current_model_performance}.")

    if os.path.exists(get_best_model_path(n=n, m=m)):
        best_model = load_best_model()
        best_model_performance = evaluate(best_model, n=n, m=m, reward=reward, runs=runs)
        print(f"The average maximum load of the best model is {best_model_performance}.")
        if current_model_performance < best_model_performance:
            torch.save(current_model.state_dict(), get_best_model_path(n=n, m=m))
            print(f"The best model has been updated to the current model.")
    else:
        torch.save(current_model.state_dict(), get_best_model_path(n=n, m=m))
        print(f"This is the first model trained with these parameters. This trained model is now saved.")


if __name__ == '__main__':
    compare(n=10, m=20)
