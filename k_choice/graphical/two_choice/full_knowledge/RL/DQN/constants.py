from math import sqrt
from k_choice.graphical.two_choice.full_knowledge.RL.DQN.neural_network import *
from k_choice.graphical.two_choice.graphs.cycle import Cycle
from k_choice.graphical.two_choice.graphs.graph_base import GraphBase

N = 16  # TODO: for N=3 test if it converges to Greedy
GRAPH = Cycle(N)
M = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPS_START = 0.2
EPS_END = 0.07
EPS_DECAY = 4000
TRAIN_EPISODES = 1000
TARGET_UPDATE_FREQ = 20
MEMORY_CAPACITY = 800
EVAL_RUNS_TRAIN = 32
EVAL_RUNS_EVAL = 100
EVAL_PARALLEL_BATCH_SIZE = 32
PATIENCE = 500
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = True
OPTIMISE_FREQ = 3 * int(sqrt(M))
NN_MODEL = GeneralNet
NN_TYPE = "general_net_hypercube"  # not actually NN_TYPE but also what graph we use


def POTENTIAL_FUN(graph: GraphBase, loads):  # TODO: try smoothing out
    # TODO: look at further away bins too
    adj_sums = [(loads[i] + sum([loads[j] for j in graph.adjacency_list[i]])) for i in range(graph.n)]
    return -max(adj_sums)


def POTENTIAL_FUN_CYCLE(graph: GraphBase, loads):  # Only works for the Cycle graph, but seems to work better for that
    adj_avgs = [(loads[i] + loads[(i + 1) % graph.n]) / 2 for i in range(graph.n)]
    return -max(adj_avgs)  # TODO: take into account more bins, not just 2


def REWARD_FUN(x):
    return -max(x)
