from unittest import TestCase
from two_thinning.strategies.mean_thinning_strategy import MeanThinningStrategy


class TestMeanThinningStrategy(TestCase):
    def __init__(self):
        super().__init__()
        self.strategy = MeanThinningStrategy(n=3, m=5)

    def test_decide(self):
        self.fail()

    def test_note(self):
        self.fail()

    def test_reset(self):
        self.fail()
