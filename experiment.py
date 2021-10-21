import torch
import numpy as np

from two_thinning.RL.two_thinning_models import TwoThinningNet

n=10
m=n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoThinningNet(n,m)
model.to(device).double()
model.load_state_dict(torch.load('two_thinning/RL/saved_models/best.pth'))
model.eval()

eval_episodes=3


max_loads=[]
for _ in range(eval_episodes):
    loads=np.zeros(n)
    for i in range(m):
        options=model(torch.from_numpy(loads).double())
        print(f"The options are: {options}")
        a=torch.argmax(options)
        print(f"With load vector {loads}, the model chooses a threshold {a}")
        randomly_selected=np.random.randint(n)
        if loads[randomly_selected]<=a:
            loads[randomly_selected]+=1
        else:
            loads[np.random.randint(n)]+=1
    max_loads.append(np.max(loads))

avg_max_load=sum(max_loads)/len(max_loads)
print(avg_max_load)