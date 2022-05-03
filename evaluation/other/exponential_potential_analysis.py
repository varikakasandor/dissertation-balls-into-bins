from math import exp
import numpy as np
from matplotlib import pyplot as plt

AVG = 2

def exponential_potential(avg, x, alpha):
    return exp(alpha*(x-avg))


def create_analyses(avg=AVG):
    for alpha in list(reversed(list(np.linspace(1, 0, 4, endpoint=False)))):
        xs = list(range(8))
        ys = [exponential_potential(avg, x, alpha) for x in xs]
        plt.plot(xs, ys, label=f"alpha={alpha}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("exp(alpha*(x-2))")
    plt.savefig("exponential_potential_analysis.png")
    plt.savefig("exponential_potential_analysis.pdf")


if __name__=="__main__":
    create_analyses()