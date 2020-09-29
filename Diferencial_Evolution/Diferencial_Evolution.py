import numpy as np
# import matplotlib.pyplot as plt
import time
from scipy.optimize import differential_evolution as d_e


def F(x):
    # to find the maximum of this function
    return np.sin(10*x)*x + np.cos(2*x)*x


# def fobj(x):
#   value = 0
#   for i in range(len(x)):
#       value += x[i]**2
#   return value / len(x)


def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


def main():

    # fobj = lambda x: sum(x ** 2) / len(x)

    # it = list(de(lambda x: x ** 2, bounds=[(-100, 100)]))
    # it = list(de(lambda x: np.sin(10*x)*x + np.cos(2*x)*x, bounds=[(-5, 5)]))

    # r = list(de(lambda x: x ** 2 / len(x), bounds=[(-100, 100)] * 32))

    # compute the maximum (-f)
    print('\nOwn function:')
    start_time = time.time_ns()
    result = list(
        de(lambda x: -(np.sin(10*x)*x + np.cos(2*x)*x), bounds=[(-5, 5)]))
    print("Time =", (time.time_ns() - start_time)/1e6, "ms")
    x, f = zip(*result)
    print("The best solution x={0} with f(x)={1}".format(x[-1][0], -f[-1][0]))

    print('\nscipy function:')
    start_time = time.time_ns()
    result = d_e(lambda x: -(np.sin(10*x)*x + np.cos(2*x)*x), bounds=[(-5, 5)])
    print("Time =", (time.time_ns() - start_time)/1e6, "ms")
    # x, f = zip(*result)
    print("The best solution x={0} with f(x)={1}".format(
        result['x'][0], -result['fun'][0]))

    # plt.plot(f)
    # plt.show()


if __name__ == "__main__":
    main()
