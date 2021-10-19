import random

n=100
m=200
episodes=10000
epsilon=0.1

def reward(threshold): #Should be imported from two_thinning_simulation
    loads=[0]*n
    for _ in range(m):
        chosen=random.randrange(n)
        if loads[chosen]<=threshold:
            loads[chosen]+=1
        else:
            arbitrary=random.randrange(n)
            loads[arbitrary]+=1
    return max(loads)


def train(n=n, m=m, episodes=episodes, epsilon=epsilon):
    q=[0]*(n+1)
    cnt=[0]*(n+1)
    for _ in range(episodes):
        r=random.random()
        if r<epsilon:
            a=random.randrange(n+1)
        else:
            a=q.index(min(q))
        r=reward(a)
        cnt[a]+=1
        q[a]+=(r-q[a])/cnt[a]
    print(q)

if __name__=="__main__":
    train()
