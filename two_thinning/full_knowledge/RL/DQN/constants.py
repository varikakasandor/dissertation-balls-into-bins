from math import sqrt
from two_thinning.full_knowledge.RL.DQN.neural_network import *

N = 10
M = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPS_START = 0.2
EPS_END = 0.05
EPS_DECAY = 2000
CONTINUOUS_REWARD = True
TRAIN_EPISODES = 1000
TARGET_UPDATE_FREQ = 10
MEMORY_CAPACITY = 10 * BATCH_SIZE
EVAL_RUNS_TRAIN = 5
EVAL_RUNS_EVAL = 10
EVAL_PARALLEL_BATCH_SIZE = 32
PATIENCE = 200
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = True
OPTIMISE_FREQ = int(sqrt(M))  # TODO: completely ad-hoc
MAX_THRESHOLD = max(3, 2 * (M + N - 1) // N)
NN_MODEL = FullTwoThinningRecurrentNetFC
NN_TYPE = "rnn_fc"


def POTENTIAL_FUN(loads):
    return -max(loads)  # TODO: take into account more bins


def REWARD_FUN(loads):  # TODO: Not yet used in training, it is hardcoded
    return -max(loads)
