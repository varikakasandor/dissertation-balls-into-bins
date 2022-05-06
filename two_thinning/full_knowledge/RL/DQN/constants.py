from math import sqrt, log, ceil, exp
from os.path import join, dirname, abspath
from datetime import datetime

from two_thinning.full_knowledge.RL.DQN.neural_network import *
from helper.helper import std

N = 20
M = 400


def NO_POTENTIAL(loads):
    return 0


def EXPONENTIAL_POTENTIAL(loads, alpha=0.5):
    t = sum(loads)
    n = len(loads)
    potential = sum([exp(alpha * (x - t / n)) for x in loads])
    return -potential


def STD_POTENTIAL(loads):
    return -std(loads)


def MAX_LOAD_POTENTIAL(loads):
    return -max(loads)


def MAX_LOAD_REWARD(loads):
    return -max(loads)


def CORRECTED_MAX_LOAD_REWARD(loads, error_ratio=1.5):
    return 1 if max(loads) < error_ratio * sum(loads) / len(loads) else 0


def EVEN_PACING_FUN(start_size, n=N, m=M, all_episodes=1000):
    # 1 + (1+x) + (1+2x) + ... + (1+(m-1)x) = all_episodes, that is
    # m + x*m*(m-1)/2 = all_episodes, giving (changing x to delta, adding max)
    delta = max((all_episodes-m) * 2 // (m * (m - 1)), 0)
    if delta == 0:
        return 1 if start_size % max((m // all_episodes), 1) == 0 else 0  # make sure to have at most all_episodes overall
    return 1 + start_size * delta  # later choices are more important!


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPS_START = 0.2
EPS_END = 0.05
EPS_DECAY = 3500
TRAIN_EPISODES = 350
PRE_TRAIN_EPISODES = 50
TARGET_UPDATE_FREQ = 20
MEMORY_CAPACITY = 450
EVAL_RUNS_TRAIN = 5
EVAL_RUNS_EVAL = 1000
EVAL_PARALLEL_BATCH_SIZE = 64
PATIENCE = 1000
USE_NORMALISED = True
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = True
OPTIMISE_FREQ = 25
MAX_THRESHOLD = 22
NN_MODEL = GeneralNet
NN_TYPE = "general_net"
LOSS_FUCNTION = nn.MSELoss()
LR = 0.001
NN_HIDDEN_SIZE = 128
NN_RNN_NUM_LAYERS = 3
NN_NUM_LIN_LAYERS = 2
OPTIMIZER_METHOD = torch.optim.Adam
SAVE_PATH = join(dirname(dirname(dirname(dirname(dirname(abspath(__file__)))))), "evaluation", "training_progression",
                 f'{str(datetime.now().strftime("%Y_%m_%d %H_%M_%S_%f"))}_{N}_{M}')
REWARD_FUN = MAX_LOAD_REWARD
POTENTIAL_FUN = MAX_LOAD_POTENTIAL
PACING_FUN = EVEN_PACING_FUN
