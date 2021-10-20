import functools
import random
import time

from mathematical_results import two_thinning_constant_threshold_maths
from helper import number_of_increasing_partitions


n=20
m=n

@functools.lru_cache(maxsize=n*m*number_of_increasing_partitions(m, n))
def two_thinning_constant_threshold_dp(loads_tuple, chosen, threshold):
    loads=list(loads_tuple)
    if sum(loads)==m:
        return max(loads)
    elif loads[chosen]<=threshold:
        loads[chosen]+=1
        res=0
        loads_sorted=sorted(loads)
        for i in range(n):
            res+=two_thinning_constant_threshold_dp(tuple(loads_sorted), i, threshold)
        return res/n
    else:
        res=0
        for i in range(n):
            loads[i]+=1
            loads_sorted=sorted(loads)
            for j in range(n):
                res+=two_thinning_constant_threshold_dp(tuple(loads_sorted), j, threshold)
            loads[i]-=1
        return res/(n*n)

@functools.lru_cache(maxsize=400000)
def two_thinning_constant_threshold_simpler_state_dp(loads_tuple, threshold):
    loads=list(loads_tuple)
    if sum(loads)==m:
        return max(loads)
    else:
        res=0
        for chosen in range(n):
            if loads[chosen]<=threshold:
                loads[chosen]+=1
                res+=two_thinning_constant_threshold_simpler_state_dp(tuple(loads), threshold)
                loads[chosen]-=1
            else:
                subres=0
                for rejected in range(n):
                    loads[rejected]+=1
                    subres+=two_thinning_constant_threshold_simpler_state_dp(tuple(loads), threshold)
                    loads[rejected]-=1
                res+=subres/n
        return res/n


@functools.lru_cache(maxsize=m*n*number_of_increasing_partitions(m, n))
def two_thinning_dp(loads_tuple, chosen):
    loads=list(loads_tuple)
    if sum(loads)==m:
        return max(loads)

    accept=0
    loads[chosen]+=1
    loads_sorted=sorted(loads)
    for i in range(n):
        accept+=two_thinning_dp(tuple(loads_sorted), i)
    loads[chosen]-=1
    accept/=n

    reject=0
    for i in range(n):
        loads[i]+=1
        loads_sorted=sorted(loads)
        for j in range(n):
            reject+=two_thinning_dp(tuple(loads_sorted), j)
        loads[i]-=1
    reject/=(n*n)


    return min(accept, reject)


if __name__=="__main__":
    start_time = time.time()
    for threshold in range(m+1):
        print(f"With {m} balls, {n} bins and constant threshold {threshold} the expected maximum load is {two_thinning_constant_threshold_dp(tuple([0]*n),0,threshold)}")

    print(f"With {m} balls, {n} bins the best achievable expected maximum load is {two_thinning_dp(tuple([0]*n),0)}")
    #print(f"According to the mathematical bounds the optimal constant threshold should be around {two_thinning_constant_threshold_maths(n)}")
    print("--- %s seconds ---" % (time.time() - start_time))
