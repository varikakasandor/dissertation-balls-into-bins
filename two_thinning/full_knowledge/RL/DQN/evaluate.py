import os

from two_thinning.full_knowledge.RL.DQN.constants import *
from two_thinning.full_knowledge.RL.DQN.neural_network import FullTwoThinningOneHotNet, FullTwoThinningNet, \
    FullTwoThinningRecurrentNet
from two_thinning.full_knowledge.RL.DQN.train import train, evaluate_q_values_faster


def get_best_model_path(n=N, m=M, nn_type=NN_TYPE):
    nn_type_str = f"{nn_type}_" if nn_type is not None else ""
    best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models",
                                   f"best_{nn_type_str}{n}_{m}.pth")
    return best_model_path


def load_best_model(n=N, m=M, nn_type=NN_TYPE, device=DEVICE):
    model = FullTwoThinningOneHotNet if nn_type == "one_hot" else (
        FullTwoThinningRecurrentNet if nn_type == "rnn" else (
            FullTwoThinningRecurrentNetFC if nn_type == "rnn_fc" else FullTwoThinningNet))

    for max_threshold in range(m + 1):
        try:
            best_model = model(n=n, max_threshold=max_threshold, max_possible_load=m,
                               device=device)
            best_model.load_state_dict(torch.load(get_best_model_path(n=n, m=m, nn_type=nn_type)))
            best_model.eval()
            return best_model
        except:
            continue

    print("ERROR: trained model not found with any max_threshold")
    return None


def evaluate(trained_model, n=N, m=M, reward_fun=REWARD_FUN, eval_runs_eval=EVAL_RUNS_EVAL,
             eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE):
    avg_score = evaluate_q_values_faster(trained_model, n=n, m=m, reward=reward_fun, eval_runs=eval_runs_eval,
                                         batch_size=eval_parallel_batch_size)  # TODO: set back print_behaviour to True
    return avg_score


def compare(n=N, m=M, train_episodes=TRAIN_EPISODES, memory_capacity=MEMORY_CAPACITY, eps_start=EPS_START,
            eps_end=EPS_END, eps_decay=EPS_DECAY, reward_fun=REWARD_FUN, batch_size=BATCH_SIZE,
            optimise_freq=OPTIMISE_FREQ, eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE,
            target_update_freq=TARGET_UPDATE_FREQ, max_threshold=MAX_THRESHOLD,
            eval_runs_train=EVAL_RUNS_TRAIN, eval_runs_eval=EVAL_RUNS_EVAL, patience=PATIENCE,
            print_progress=PRINT_PROGRESS, device=DEVICE, nn_model=NN_MODEL, potential_fun=POTENTIAL_FUN,
            nn_type=NN_TYPE):
    current_model = train(n=n, m=m, memory_capacity=memory_capacity, num_episodes=train_episodes, reward_fun=reward_fun,
                          batch_size=batch_size, eps_start=eps_start, eps_end=eps_end,
                          max_threshold=max_threshold, optimise_freq=optimise_freq, potential_fun=potential_fun,
                          eps_decay=eps_decay, target_update_freq=target_update_freq, eval_runs=eval_runs_train,
                          patience=patience, print_progress=print_progress, nn_model=nn_model,
                          device=device, eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE)
    current_model_performance = evaluate(current_model, n=n, m=m, reward_fun=reward_fun, eval_runs_eval=eval_runs_eval,
                                         eval_parallel_batch_size=eval_parallel_batch_size)
    print(
        f"With {m} balls and {n} bins the trained current DQN model has an average score/maximum load of {current_model_performance}.")

    if os.path.exists(get_best_model_path(n=n, m=m, nn_type=nn_type)):
        best_model = load_best_model(n=n, m=m, nn_type=nn_type, device=device)
        best_model_performance = evaluate(best_model, n=n, m=m, reward_fun=reward_fun, eval_runs_eval=eval_runs_eval,
                                          eval_parallel_batch_size=eval_parallel_batch_size)
        print(f"The average score/maximum load of the best model is {best_model_performance}.")
        if current_model_performance > best_model_performance:
            torch.save(current_model.state_dict(), get_best_model_path(n=n, m=m, nn_type=nn_type))
            print(f"The best model has been updated to the current model.")
        else:
            print(f"The best model had better performance than the current one, so it is not updated.")
    else:
        torch.save(current_model.state_dict(), get_best_model_path(n=n, m=m, nn_type=nn_type))
        print(f"This is the first model trained with these parameters. This trained model is now saved.")


if __name__ == "__main__":
    compare()
