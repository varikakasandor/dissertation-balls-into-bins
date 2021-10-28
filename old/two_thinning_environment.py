import numpy as np
import gym
from gym import spaces
import random

class TwoThinning(gym.Env):
    REJECT = 0
    ACCEPT = 1

    def get_random_bin(self):
        return random.randrange(self.n)

    def evaluate(self):
        return -np.max(self.load_configuration)

    def __init__(self, n=8, m=15):
        super(TwoThinning, self).__init__()

        self.n=n
        self.m=m
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({"load_configuration": spaces.Box(low=np.array([0]*n), high=np.array([m]*n), dtype=np.float64), "location": spaces.Discrete(n)})

    def reset(self):
        self.load_configuration=np.array([0]*self.n).astype(np.float64)
        self.currently_chosen=self.get_random_bin()
        return {"load_configuration": self.load_configuration.copy(), "location": self.currently_chosen}

    def step(self, action):
        if action == self.ACCEPT:
            self.load_configuration[self.currently_chosen]+=1
        elif action == self.REJECT:
            arbitrary=self.get_random_bin()
            self.load_configuration[arbitrary]+=1


        if np.sum(self.load_configuration)==self.m:
            return {"load_configuration": self.load_configuration.copy(), "location": 0}, self.evaluate(), True, {}
        else:
            self.currently_chosen=self.get_random_bin()
            return {"load_configuration": self.load_configuration.copy(), "location": self.currently_chosen}, 0, False, {}
