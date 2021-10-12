from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO


from two_thinning_environment import TwoThinning

n=10

def test_agent(env, model, runs=20):
    rewards=[]
    for _ in range(runs):
        obs = env.reset()
        for _ in range(n):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            if done:
                rewards.append(reward)

    return sum(rewards)/len(rewards)


if __name__=="__main__":
    env = TwoThinning(n=n)
    check_env(env, warn=True)
    model = PPO('MultiInputPolicy', env).learn(500)
    avg_reward=test_agent(env, model)
    print(f"The simulated average maximum load is {-avg_reward}")