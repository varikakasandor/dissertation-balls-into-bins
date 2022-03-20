from k_choice.graphical.two_choice.graph_base import GraphBase


class HyperCube(GraphBase):
    def __init__(self, n):
        self._n = 2 ** n
        self._e = n * (2 ** (n-1))
        self._d = n
        self._adjacency_list = [[(i ^ (2 ** j)) for j in range(n)] for i in range(2 ** n)]
        self._edge_list = [(i, j) for i, sublist in enumerate(self._adjacency_list) for j in sublist if j >= i]

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
