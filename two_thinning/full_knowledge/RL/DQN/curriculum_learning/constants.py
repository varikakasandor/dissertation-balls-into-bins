from math import sqrt, log, ceil
from two_thinning.full_knowledge.RL.DQN.neural_network import *
from helper.helper import std

N = 5
M = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPS_START = 0.2
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE_FREQ = 10
MEMORY_CAPACITY = 10 * BATCH_SIZE
EVAL_RUNS_TRAIN = 10
EVAL_RUNS_EVAL = 100
EVAL_PARALLEL_BATCH_SIZE = 32
PATIENCE = 1000
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = True
OPTIMISE_FREQ = int(sqrt(M))  # TODO: completely ad-hoc
MAX_THRESHOLD = max(3, M // N + ceil(sqrt(log(N))))
NN_MODEL = FullTwoThinningClippedRecurrentNetFC
NN_TYPE = "curriculum_rnn_clipped_fc"


def POTENTIAL_FUN(loads):
    return -max(loads)  # TODO: take into account more bins
    # return -std(loads)


def REWARD_FUN(loads, error_ratio=1.5):
    return -max(loads)
    # return -std(loads)
    # return 1 if max(loads) < error_ratio * sum(loads) / len(loads) else 0


def PACING_FUN(start_size, n=N, m=M,
               all_epochs=1000):  # returns number of epochs to run for the given start size. Note that the exact
    # definition of "pacing function" is different
    # I assume we want linearly increasing pacing function, so we need to solve for x and that we start from n:
    # n + (n+x) + (n+2x) + ... + (n+(m-1)x) = all_epochs, that is
    # n*m + x*m*(m-1)/2 = all_epochs, giving (changing x to delta, adding max)
    delta = max((all_epochs - n * m) * 2 // (m * (m - 1)), 0)
    return n + ((m - 1) - start_size) * delta
