from math import sqrt, log, ceil, exp
from os.path import join, dirname, abspath
from datetime import datetime

import torch.optim

from k_thinning.full_knowledge.RL.DQN.neural_network import *
from helper.helper import std

N = 5
M = 25
K = 3



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
BATCH_SIZE = 21
EPS_START = 0.106
EPS_END = 0.035
EPS_DECAY = 4073
TRAIN_EPISODES = 550
PRE_TRAIN_EPISODES = 50
TARGET_UPDATE_FREQ = 23
MEMORY_CAPACITY = 905
EVAL_RUNS_TRAIN = 8
EVAL_RUNS_EVAL = 100
EVAL_PARALLEL_BATCH_SIZE = 16  # 64
PATIENCE = 1000
USE_NORMALISED = False
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = True
OPTIMISE_FREQ = 15  # TODO: 50 for N=10, M=100
MAX_THRESHOLD = 7  #max(3, M // N + ceil(
    #sqrt(log(N))))  # Should be given in absolute domain even if normalised domain if USE_NORMALISED=True
NN_MODEL = GeneralNet
NN_TYPE = "general_net"
LOSS_FUCNTION = nn.SmoothL1Loss()
LR = 0.001  # 0.0028
NN_HIDDEN_SIZE = 187
NN_RNN_NUM_LAYERS = 1
NN_NUM_LIN_LAYERS = 1
OPTIMIZER_METHOD = torch.optim.Adagrad
SAVE_PATH = join(dirname(dirname(dirname(dirname(dirname(abspath(__file__)))))), "evaluation", "training_progression",
                 f'{str(datetime.now().strftime("%Y_%m_%d %H_%M_%S_%f"))}_{N}_{M}')
REWARD_FUN = MAX_LOAD_REWARD
POTENTIAL_FUN = STD_POTENTIAL
PACING_FUN = EVEN_PACING_FUN
