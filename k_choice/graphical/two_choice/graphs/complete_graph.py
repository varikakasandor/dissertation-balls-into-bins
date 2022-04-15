from k_choice.graphical.two_choice.graphs.graph_base import GraphBase


class CompleteGraph(GraphBase):
    def __init__(self, n):
        self._n = n
        self._e = n * (n + 1) // 2
        self._d = n
        self._adjacency_list = [list(range(n)) for _ in range(n)]
        self._edge_list = [(i, j) for i in range(n) for j in range(i, n)]

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
