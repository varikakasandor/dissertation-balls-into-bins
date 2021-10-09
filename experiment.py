import functools
import random

n=8

@functools.lru_cache(maxsize=400000)
def dp_two_thinning_constant_threshold(loads_tuple, chosen, threshold):
    loads=list(loads_tuple)
    if sum(loads)==n:
        return max(loads)
    elif loads[chosen]<=threshold:
        loads[chosen]+=1
        res=0
        for i in range(n):
            res+=dp_two_thinning_constant_threshold(tuple(loads), i, threshold)
        return res/n
    else:
        res=0
        for i in range(n):
            loads[i]+=1
            for j in range(n):
                res+=dp_two_thinning_constant_threshold(tuple(loads), j, threshold)
            loads[i]-=1
        return res/(n*n)


if __name__=="__main__":
    for threshold in range(n+1):
        print(f"With threshold {threshold} the expected maximum load is {dp_two_thinning_constant_threshold(tuple([0]*n),0,threshold)}")