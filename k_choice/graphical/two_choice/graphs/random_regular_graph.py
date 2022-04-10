import random

from k_choice.graphical.two_choice.graph_base import GraphBase
from math import log2


class RandomRegularGraph(GraphBase):
    def __init__(self, n, d):
        assert 0 <= d <= n - 1 and (n * d) % 2 == 0
        self._n = n
        self._e = n * d // 2
        self._d = d
        self._adjacency_list = RandomRegularGraph.create_random_regular_graph(n, d)
        self._edge_list = [(i, j) for i in range(n) for j in self._adjacency_list[i] if j > i]

    def create_random_regular_graph(n, d):
        while True:
            adj = [[] for _ in range(n)]
            S = [(i, j) for i in range(n) for j in range(i + 1, n)]
            while True:
                weights = [(d - len(adj[i])) * (d - len(adj[j])) if (
                            len(adj[i]) < d and len(adj[j]) < d and j not in adj[i]) else 0 for (i, j) in S]
                if sum(weights) == 0:
                    break
                (x, y) = random.choices(S, weights=weights)[0]
                adj[x].append(y)
                adj[y].append(x)
                S.remove((x, y))
            if sum([len(l) for l in adj]) == d * n:
                break
        return adj

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
