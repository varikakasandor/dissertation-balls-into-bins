from math import sqrt
from k_choice.graphical.two_choice.full_knowledge.RL.DQN.neural_network import *
from k_choice.graphical.two_choice.graphs.cycle import Cycle
from k_choice.graphical.two_choice.graph_base import GraphBase

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
OPTIMISE_FREQ = int(sqrt(M))
NN_MODEL = FullGraphicalTwoChoiceOneHotFCNet
NN_TYPE = "fc_one_hot_cycle"


def POTENTIAL_FUN_GENERAL(graph: GraphBase, loads):
    adj_sums = [(loads[i] + sum([loads[j] for j in graph.adjacency_list[i]])) for i in range(graph.n)]
    return -max(adj_sums)


def POTENTIAL_FUN(graph: GraphBase, loads):  # Only works for the Cycle graph, but seems to work better for that
    adj_avgs = [(loads[i] + loads[(i + 1) % graph.n]) / 2 for i in range(graph.n)]
    return -max(adj_avgs)  # TODO: take into account more bins, not just 2


def REWARD_FUN(x):  # TODO: Not yet used in training, it is hardcoded
    return -max(x)
