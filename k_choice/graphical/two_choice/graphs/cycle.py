from k_choice.graphical.two_choice.graphs.graph_base import GraphBase


class Cycle(GraphBase):
    def __init__(self, n):
        self._n = n
        self._e = n
        self._d = 2
        self._adjacency_list = [[((i + 1) % n), ((i + n - 1) % n)] for i in range(n)]
        self._edge_list = [(i, (i + 1) % n) for i in range(n)]

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
