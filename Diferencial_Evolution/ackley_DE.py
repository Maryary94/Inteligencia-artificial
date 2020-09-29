# ackley
#
# 20 + exp(1) - 20*exp((-0.2)*sqrt(0.5*x^2))
# 20 + np.exp(1) - 20 *np.exp((-0.2)* np.sqrt((0.5)*(x**2)))

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution as d_e


def F(x):
    # to find the maximum of this function
    return 20 + np.exp(1) - 20 * np.exp((-0.2) * np.sqrt((0.5)*(x1**2 + x2**2)))


def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)  #
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind, ind) for ind in pop_denorm])
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
            f = fobj(trial_denorm, trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


def main():
    arquivo = open("Diferencial_Evolution/ackley_DE.txt", "w")
    arquivo.write(
        "Algoritmo de Diferencial Evolution com a funcao de Ackley\n")
    arquivo.write("N.Iteracoes \tSolucao\n")

    for i in range(100):

        start_time = time.time_ns()
        result = list(de(lambda x1, x2: (20 + np.exp(1) - 20 * np.exp((-0.2)
                                                                      * np.sqrt((0.5)*(x1**2 + x2**2)))), bounds=[(-5, 5)]))

        print("\nTime =", (time.time_ns() - start_time)/1e6, "ms")

        x, f = zip(*result)
        arquivo.write("Iteracao: " + str(i)+"\t\t\t\t")
        arquivo.write(
            "\nTime: " + str((time.time_ns() - start_time)/1e6) + "ms\n")

        arquivo.write(
            "x1={0} \t\t\t\t\t\t\t with f(x1)={1}\n".format(x[-1][0], f[-1][0]))
        arquivo.write(
            "x2={0} \t\t with f(x2)={1}\n\n".format(x[0][0], f[0][0]))

    arquivo.close


if __name__ == "__main__":
    main()
