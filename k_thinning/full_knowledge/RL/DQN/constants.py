from math import sqrt, log, ceil, exp
from os.path import join, dirname, abspath
from datetime import datetime

from k_thinning.full_knowledge.RL.DQN.neural_network import *
from helper.helper import std

N = 5
M = 35
K = 3


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


def EVEN_PACING_FUN(start_size, n=N, m=M,
                    all_episodes=1000):  # returns number of episodes to run for the given start size. Note that the exact
    # definition of "pacing function" is different
    # I assume we want linearly increasing pacing function, so we need to solve for x and that we start from n:
    # n + (n+x) + (n+2x) + ... + (n+(m-1)x) = all_episodes, that is
    # n*m + x*m*(m-1)/2 = all_episodes, giving (changing x to delta, adding max)
    delta = max((all_episodes - n * m) * 2 // (m * (m - 1)), 0)
    return n + ((m - 1) - start_size) * delta


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPS_START = 0.2
EPS_END = 0.04
EPS_DECAY = 4200
TRAIN_EPISODES = 70
PRE_TRAIN_EPISODES = 30
TARGET_UPDATE_FREQ = 20
MEMORY_CAPACITY = 800
EVAL_RUNS_TRAIN = 1
EVAL_RUNS_EVAL = 100
EVAL_PARALLEL_BATCH_SIZE = 16  # 64
PATIENCE = 5000
USE_NORMALISED = True
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = True
OPTIMISE_FREQ = 3 * int(sqrt(M))  # TODO: 50 for N=10, M=100
MAX_THRESHOLD = max(3, M // N + ceil(
    sqrt(log(N))))  # Should be given in absolute domain even if normalised domain if USE_NORMALISED=True
NN_MODEL = GeneralNet
NN_TYPE = "general_net"
LOSS_FUCNTION = nn.SmoothL1Loss()
LR = 0.0004
NN_HIDDEN_SIZE = 64
NN_RNN_NUM_LAYERS = 1
NN_NUM_LIN_LAYERS = 1
OPTIMIZER_METHOD = torch.optim.Adam
SAVE_PATH = join(dirname(dirname(dirname(dirname(dirname(abspath(__file__)))))), "evaluation", "training_progression",
                 f'{str(datetime.now().strftime("%Y_%m_%d %H_%M_%S_%f"))}_{N}_{M}')
REWARD_FUN = MAX_LOAD_REWARD
POTENTIAL_FUN = MAX_LOAD_POTENTIAL
PACING_FUN = EVEN_PACING_FUN
