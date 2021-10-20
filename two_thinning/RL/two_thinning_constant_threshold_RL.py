import random

n=2
m=1000
episodes=10000
epsilon=0.1

def reward(threshold, n=n, m=m): #Should be imported from two_thinning_simulation
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
    q=[0]*(m+1)
    cnt=[0]*(m+1)
    for _ in range(episodes):
        r=random.random()
        if r<epsilon:
            a=random.randrange(m+1)
        else:
            a=q.index(min(q))
        r=reward(a,n,m)
        cnt[a]+=1
        q[a]+=(r-q[a])/cnt[a]
    
    best_threshold=q.index(min(q))
    print(f"The best threshold is {best_threshold}, producing on average a maximum load of {q[best_threshold]}")

if __name__=="__main__":
    train()
