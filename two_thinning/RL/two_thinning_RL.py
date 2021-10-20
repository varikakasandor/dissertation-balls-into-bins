import random

n=20
m=n
episodes=100000
epsilon=0.01
alpha=0.1

def reward(loads):
    return max(loads)

def train(n=n, m=m, episodes=episodes, epsilon=epsilon):
    q=[[0]*(i+1) for i in range(m)]
    for _ in range(episodes):
        loads=[0]*n
        for i in range(m):
            r=random.random()
            if r<epsilon:
                a=random.randrange(i+1)
            else:
                a=q[i].index(min(q[i]))
            randomly_selected=random.randrange(n)
            if loads[randomly_selected]<=a:
                loads[randomly_selected]+=1
            else:
                loads[random.randrange(n)]+=1

            if i==m-1:
                q[i][a]+=alpha*(reward(loads)-q[i][a])
            else:
                q[i][a]+=alpha*(min(q[i+1])-q[i][a])
    
    for i in range(m):
        print(f"After {i} balls have been placed, the ideal threshold is {q[i].index(min(q[i]))} with an expected maximum load of {min(q[i])}")
    print(q[-1])

if __name__=="__main__":
    train()
