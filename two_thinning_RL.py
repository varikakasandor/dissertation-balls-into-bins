from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, A2C


from two_thinning_environment import TwoThinning

if __name__=="__main__":
    env = TwoThinning()
    check_env(env, warn=True)
    model = DQN('MlpPolicy', env, verbose=1).learn(5000)