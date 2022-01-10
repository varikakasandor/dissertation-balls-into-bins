from math import sqrt
from k_thinning.full_knowledge.RL.DQN.neural_network import *

N = 200
M = 300
K = 3

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
EVAL_RUNS_EVAL = 100
PATIENCE = 200
MAX_LOAD_INCREASE_REWARD = -1
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = True
OPTIMISE_FREQ = int(sqrt(M))  # TODO: completely ad-hoc
MAX_THRESHOLD = max(3, 2 * (M + N - 1) // N)
NN_MODEL = FullKThinningRecurrentNet
NN_TYPE = "rnn"


def REWARD_FUN(x):  # TODO: Not yet used in training, it is hardcoded
    return -max(x)
