from math import sqrt, log, ceil
from two_thinning.full_knowledge.RL.DQN.neural_network import *
from helper.helper import std

N = 10
M = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPS_START = 0.2
EPS_END = 0.05
EPS_DECAY = 2000
TRAIN_EPISODES = 3000
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
NN_TYPE = "rnn_clipped_fc"


def POTENTIAL_FUN(loads):
    return -max(loads)  # TODO: take into account more bins
    #return -std(loads)

def REWARD_FUN(loads, error_ratio=1.5):
    return -max(loads)
    #return -std(loads)
    #return 1 if max(loads) < error_ratio * sum(loads) / len(loads) else 0
