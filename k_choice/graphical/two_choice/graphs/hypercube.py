from k_choice.graphical.two_choice.graphs.graph_base import GraphBase
from math import log2


class HyperCube(GraphBase):
    def __init__(self, n):
        assert n != 0 and (n & (n - 1) == 0)  # has to be a power of 2
        num_bits = int(log2(n))
        self._n = n
        self._e = num_bits * (n // 2)
        self._d = num_bits
        self._adjacency_list = [[(i ^ (2 ** j)) for j in range(num_bits)] for i in range(n)]
        self._edge_list = [(i, j) for i, sublist in enumerate(self._adjacency_list) for j in sublist if j >= i]

    @property
    def name(self):
        return "hypercube"

    @property
    def n(self):
        return self._n

    @property
    def e(self):
        return self._e

    @property
    def d(self):
        return self._d

    @property
    def adjacency_list(self):
        return self._adjacency_list

    @property
    def edge_list(self):
        return self._edge_list
