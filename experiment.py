import functools
import random

n=9

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

@functools.lru_cache(maxsize=400000)
def dp_two_thinning_constant_threshold_simpler_state(loads_tuple, threshold):
    loads=list(loads_tuple)
    if sum(loads)==n:
        return max(loads)
    else:
        res=0
        for chosen in range(n):
            if loads[chosen]<=threshold:
                loads[chosen]+=1
                res+=dp_two_thinning_constant_threshold(tuple(loads), threshold)
                loads[chosen]-=1
            else:
                subres=0
                for rejected in range(n):
                    loads[rejected]+=1
                    subres+=dp_two_thinning_constant_threshold(tuple(loads), threshold)
                    loads[rejected]-=1
                res+=subres/n
        return res/n


@functools.lru_cache(maxsize=400000)
def dp_two_thinning(loads_tuple, chosen):
    loads=list(loads_tuple)
    if sum(loads)==n:
        return max(loads)

    accept=0
    loads[chosen]+=1
    for i in range(n):
        accept+=dp_two_thinning(tuple(loads), i)
    loads[chosen]-=1
    accept/=n

    reject=0
    for i in range(n):
        loads[i]+=1
        for j in range(n):
            reject+=dp_two_thinning(tuple(loads), j)
        loads[i]-=1
    reject/=(n*n)


    return min(accept, reject)


if __name__=="__main__":
    for threshold in range(n+1):
        print(f"With {n} balls, {n} bins and constant threshold {threshold} the expected maximum load is {dp_two_thinning_constant_threshold(tuple([0]*n),0,threshold)}")

    print(f"With {n} balls, {n} bins the best achievable expected maximum load is {dp_two_thinning(tuple([0]*n),0)}")