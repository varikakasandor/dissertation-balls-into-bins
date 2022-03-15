from math import sqrt, log, ceil
from two_thinning.full_knowledge.RL.DQN.neural_network import *
from helper.helper import std

N = 5
M = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPS_START = 0.2
EPS_END = 0.07
EPS_DECAY = 5000
TRAIN_EPISODES = 3000
TARGET_UPDATE_FREQ = 20
MEMORY_CAPACITY = 10 * BATCH_SIZE
EVAL_RUNS_TRAIN = 32
EVAL_RUNS_EVAL = 100
EVAL_PARALLEL_BATCH_SIZE = 32
PATIENCE = 400
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = True
OPTIMISE_FREQ = 5 * int(sqrt(M))  # TODO: 50 for N=10, M=100
MAX_THRESHOLD = max(3, M // N + ceil(sqrt(log(N))))
NN_MODEL = FullTwoThinningClippedRecurrentNetFC
NN_TYPE = "rnn_clipped_fc"


def POTENTIAL_FUN(loads):
    return -max(loads)  # TODO: take into account more bins
    #return -std(loads)

def REWARD_FUN(loads, error_ratio=1.5):
    return -max(loads)
    #return -std(loads)
    #return 1 if max(loads) < error_ratio * sum(loads) / len(loads) else 0
