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

    def __init__(self, n=10):
        super(TwoThinning, self).__init__()

        self.n=n
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({"load_configuration": spaces.Box(low=np.array([0]*n), high=np.array([n]*n), dtype=np.uint), "location": spaces.Discrete(n)})

    def reset(self):
        self.load_configuration=np.array([0]*self.n).astype(np.uint)
        self.currently_chosen=self.get_random_bin()
        return {"load_configuration": self.load_configuration, "location": self.currently_chosen}

    def step(self, action):
        if action == self.ACCEPT:
            self.load_configuration[self.currently_chosen]+=1
        elif action == self.REJECT:
            arbitrary=self.get_random_bin()
            self.load_configuration[arbitrary]+=1


        if np.sum(self.load_configuration)==self.n:
            return {"load_configuration": self.load_configuration, "location": 0}, self.evaluate(), True, {}
        else:
            self.currently_chosen=self.get_random_bin()
            return {"load_configuration": self.load_configuration, "location": self.currently_chosen}, 0, False, {}


    def close(self):
        pass
    