from abc import ABCMeta, abstractmethod
from copy import deepcopy


class GraphBase(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def n(self):
        pass

    @property
    @abstractmethod
    def e(self):
        pass

    @property
    @abstractmethod
    def d(self):
        pass

    @property
    @abstractmethod
    def adjacency_list(self):
        pass

    @property
    @abstractmethod
    def edge_list(self):
        pass

    def transpose(self):
        #  NOTE: this doesn't consider self loops
        new_graph = deepcopy(self)
        new_graph._e = self.n * (self.n - self.d - 1) // 2
        new_graph._d = self.n - self.d - 1
        new_graph._edge_list = [(i, j) for i in range(self.n) for j in range(i + 1, self.n) if
                               (i, j) not in self.edge_list]
        new_graph._adjacency_list = [[j for j in range(self.n) if j != i and j not in self.adjacency_list[i]] for i in
                                    range(self.n)]

        return new_graph
