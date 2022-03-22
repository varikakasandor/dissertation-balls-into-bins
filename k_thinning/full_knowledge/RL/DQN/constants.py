from math import sqrt, ceil, log
from k_thinning.full_knowledge.RL.DQN.neural_network import *

N = 10
M = 100
K = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPS_START = 0.2
EPS_END = 0.05
EPS_DECAY = 2000
CONTINUOUS_REWARD = True
TRAIN_EPISODES = 1000
TARGET_UPDATE_FREQ = 20
MEMORY_CAPACITY = 10 * BATCH_SIZE
EVAL_RUNS_TRAIN = 10
EVAL_RUNS_EVAL = 100
PATIENCE = 200
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = True
OPTIMISE_FREQ = 3 * int(sqrt(M))
MAX_THRESHOLD = max(3, M // N + ceil(sqrt(log(N)))) # could be even smaller for larger K
NN_MODEL = FullKThinningLargerRecurrentNet
NN_TYPE = "larger_rnn"

def POTENTIAL_FUN(loads):
    return -max(loads)  # TODO: take into account more bins


def REWARD_FUN(x):
    return -max(x)
