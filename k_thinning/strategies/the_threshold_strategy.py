import random
from k_thinning.strategies.strategy_base import StrategyBase


class TheThresholdStrategy(StrategyBase):
    # Works only reasonably if m=n
    def __init__(self, n, m, k, limit=None, **kwargs):
        super(TheThresholdStrategy, self).__init__(n, m, k)
        self.threshold = limit if limit is not None else self.train(**kwargs)
        self.allocations = [[0 for _ in range(self.k - 1)] for _ in range(self.n)]

    def train(self, initial_q_value=0, episodes=10000, epsilon=0.1, reward_fun=lambda loads: -max(loads)):  # finds best threshold

        def choose_random_max(q):
            max_val = max(q)
            random_maxi = random.choice([i for i in range(len(q)) if q[i] == max_val])
            return random_maxi

        def simulate_one_run(threshold):
            loads = [0] * self.n
            allocations = [[0 for _ in range(self.k - 1)] for _ in range(self.n)]
            for _ in range(self.m):
                choices_left = self.k
                final_choice = None
                while choices_left > 1:
                    final_choice = random.randrange(self.n)
                    if allocations[final_choice][choices_left - 2] <= threshold:
                        allocations[final_choice][choices_left - 2] += 1
                        break
                    else:
                        choices_left -= 1

                if choices_left == 1:
                    final_choice = random.randrange(self.n)

                loads[final_choice] += 1

            return reward_fun(loads)

        q = [initial_q_value] * (self.m + 1)
        cnt = [0] * (self.m + 1)
        for _ in range(episodes):
            r = random.random()
            if r < epsilon:
                a = random.randrange(self.m + 1)
            else:
                a = choose_random_max(q)
            result = simulate_one_run(a)
            cnt[a] += 1
            q[a] += (result - q[a]) / cnt[a]

        best_threshold = q.index(max(q))
        return best_threshold

    def decide(self, bin):
        if self.allocations[bin][self.choices_left - 2] <= self.threshold:
            self.allocations[bin][self.choices_left - 2] += 1
            return True
        else:
            return False

    def note(self, bin):
        pass

    def reset(self):
        self.allocations = [[0 for _ in range(self.k - 1)] for _ in range(self.n)]
