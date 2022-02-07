from math import sqrt
from k_choice.graphical.two_choice.full_knowledge.RL.DQN.neural_network import *
from k_choice.graphical.two_choice.graphs.cycle import Cycle

N = 4
GRAPH = Cycle(N)
M = 11

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPS_START = 0.2
EPS_END = 0.05
EPS_DECAY = 2000
CONTINUOUS_REWARD = True
TRAIN_EPISODES = 3000
TARGET_UPDATE_FREQ = 10
MEMORY_CAPACITY = 10 * BATCH_SIZE
EVAL_RUNS_TRAIN = 5
EVAL_RUNS_EVAL = 10
EVAL_PARALLEL_BATCH_SIZE = 32
PATIENCE = 1000
MAX_LOAD_INCREASE_REWARD = -1
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = True
OPTIMISE_FREQ = int(sqrt(M))  # TODO: completely ad-hoc
NN_MODEL = FullGraphicalTwoChoiceOneHotFCNet
NN_TYPE = "fc_one_hot_cycle"



def POTENTIAL_FUN(loads): # Only works for the Cycle graph
    n = len(loads)
    adj_avgs = [(loads[i] + loads[(i + 1) % n])/2 for i in range(n)]
    return -max(adj_avgs)  # TODO: take into account more bins, not just 2


def REWARD_FUN(x):  # TODO: Not yet used in training, it is hardcoded
    return -max(x)
