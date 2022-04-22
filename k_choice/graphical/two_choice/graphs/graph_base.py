from abc import ABCMeta, abstractmethod


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
