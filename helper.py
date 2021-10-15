import functools

n=8

@functools.lru_cache()
def number_of_increasing_partitions(n, k):
    if n==0 or k==1:
        return 1
    if n>=k:
        return number_of_increasing_partitions(n,k-1)+number_of_increasing_partitions(n-k,k)
    return number_of_increasing_partitions(n,k-1)


if __name__=="__main__":
    print(number_of_increasing_partitions(n,n))