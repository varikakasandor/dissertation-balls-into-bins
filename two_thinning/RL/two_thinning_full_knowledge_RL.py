import torch
import torch.nn as nn
import numpy as np

from two_thinning_models import TwoThinningNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n=10
m=n

alpha=0.1
epsilon=0.1 #TODO: set (exponential) decay
train_episodes=100
eval_episodes=100

model=TwoThinningNet(n,m,device)
model.to(device).double()
optimizer = torch.optim.Adam(model.parameters())
MSE_loss = nn.MSELoss()


def epsilon_greedy(loads, epsilon=epsilon):
    action_values=model(loads)
    r=torch.rand(1)
    if r<epsilon:
        a=torch.randint(len(action_values), (1,))
    else:
        a=torch.argmax(action_values)

    return a, action_values[a]

for _ in range(train_episodes):
    loads=np.zeros(n)
    for i in range(m):
        a,old_val=epsilon_greedy(torch.from_numpy(loads).double())
        randomly_selected=np.random.randint(n)
        if loads[randomly_selected]<=a:
            loads[randomly_selected]+=1
        else:
            loads[np.random.randint(n)]+=1
        
        if i==m-1:
            new_val=torch.from_numpy(np.array([-np.max(loads)])).to(device)
        else:
            _, new_val=epsilon_greedy(torch.from_numpy(loads).double())
            new_val=new_val.detach()

        loss = MSE_loss(old_val, new_val)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

max_loads=[]
for _ in range(eval_episodes):
    loads=np.zeros(n)
    for i in range(m):
        a,old_val=epsilon_greedy(torch.from_numpy(loads).double(),0)
        randomly_selected=np.random.randint(n)
        if loads[randomly_selected]<=a:
            loads[randomly_selected]+=1
        else:
            loads[np.random.randint(n)]+=1
    max_loads.append(np.max(loads))

print(sum(max_loads)/len(max_loads))