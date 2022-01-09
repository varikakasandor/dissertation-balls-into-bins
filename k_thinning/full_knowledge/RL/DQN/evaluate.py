import os

from k_thinning.full_knowledge.RL.DQN.constants import *
from k_thinning.full_knowledge.RL.DQN.neural_network import FullKThinningRecurrentNet
from k_thinning.full_knowledge.RL.DQN.train import train, evaluate_q_values


def get_best_model_path(n=N, m=M, k=K, nn_type=NN_TYPE):
    nn_type_str = f"{nn_type}_" if nn_type is not None else ""
    best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models",
                                   f"best_{nn_type_str}{n}_{m}_{k}.pth")
    return best_model_path


def load_best_model(n=N, m=M, k=K, nn_type=NN_TYPE, device=DEVICE):
    model = FullKThinningRecurrentNet  # FullTwoThinningOneHotNet if nn_type == "one_hot" else (FullTwoThinningRecurrentNet if nn_type == "rnn" else FullTwoThinningNet)

    for max_threshold in range(m + 1):
        try:
            best_model = model(n=n, max_threshold=max_threshold, k=k, max_possible_load=m,
                               device=device)
            best_model.load_state_dict(torch.load(get_best_model_path(n=n, m=m, nn_type=nn_type)))
            best_model.eval()
            return best_model
        except:
            continue

    print("ERROR: trained model not found with any max_threshold")
    return None


def evaluate(trained_model, n=N, m=M, k=K, reward_fun=REWARD_FUN, eval_runs_eval=EVAL_RUNS_EVAL):
    avg_score = evaluate_q_values(trained_model, n=n, m=m, k=k, reward=reward_fun, eval_runs=eval_runs_eval,
                                  print_behaviour=False)  # TODO: set back print_behaviour to True
    return avg_score


def compare(n=N, m=M, k=K, train_episodes=TRAIN_EPISODES, memory_capacity=MEMORY_CAPACITY, eps_start=EPS_START,
            eps_end=EPS_END, eps_decay=EPS_DECAY, reward_fun=REWARD_FUN, batch_size=BATCH_SIZE,
            optimise_freq=OPTIMISE_FREQ,
            target_update_freq=TARGET_UPDATE_FREQ, continuous_reward=CONTINUOUS_REWARD, max_threshold=MAX_THRESHOLD,
            eval_runs_train=EVAL_RUNS_TRAIN, eval_runs_eval=EVAL_RUNS_EVAL, patience=PATIENCE,
            max_load_increase_reward=MAX_LOAD_INCREASE_REWARD,
            print_progress=PRINT_PROGRESS, print_behaviour=PRINT_BEHAVIOUR, device=DEVICE, nn_model=NN_MODEL,
            nn_type=NN_TYPE):
    current_model = train(n=n, m=m, k=k, memory_capacity=memory_capacity, num_episodes=train_episodes,
                          reward_fun=reward_fun,
                          batch_size=batch_size, eps_start=eps_start, eps_end=eps_end,
                          continuous_reward=continuous_reward, max_threshold=max_threshold, optimise_freq=optimise_freq,
                          eps_decay=eps_decay, target_update_freq=target_update_freq, eval_runs=eval_runs_train,
                          patience=patience, max_load_increase_reward=max_load_increase_reward,
                          print_behaviour=print_behaviour, print_progress=print_progress, nn_model=NN_MODEL,
                          device=device)
    current_model_performance = evaluate(current_model, n=n, m=m, k=k, reward_fun=reward_fun, eval_runs_eval=eval_runs_eval)
    print(
        f"With {m} balls and {n} bins the trained current DQN model has an average score/maximum load of {current_model_performance}.")

    if os.path.exists(get_best_model_path(n=n, m=m, nn_type=nn_type)):
        best_model = load_best_model(n=n, m=m, k=k, device=device)
        best_model_performance = evaluate(best_model, n=n, m=m, k=k, reward_fun=reward_fun, eval_runs_eval=eval_runs_eval)
        print(f"The average maximum load of the best model is {best_model_performance}.")
        if current_model_performance > best_model_performance:
            torch.save(current_model.state_dict(), get_best_model_path(n=n, m=m, k=k, nn_type=nn_type))
            print(f"The best model has been updated to the current model.")
        else:
            print(f"The best model had better performance than the current one, so it is not updated.")
    else:
        torch.save(current_model.state_dict(), get_best_model_path(n=n, m=m, k=k, nn_type=nn_type))
        print(f"This is the first model trained with these parameters. This trained model is now saved.")


if __name__ == "__main__":
    compare()
