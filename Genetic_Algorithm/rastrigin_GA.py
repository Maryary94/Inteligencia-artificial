"""
adapted from: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import matplotlib.pyplot as plt

INDIV_SIZE = 10          # size of the individual
POP_SIZE = 250           # population size
CROSS_RATE = 0.75         # mating probability (DNA crossover)
MUTATION_RATE = 0.006    # mutation probability
N_GENERATIONS = 100      # number of generations/iterations
X_BOUND = [-5.12, 5.12]         # x upper and lower bounds


def F(x):
    # to find the maximum of this function
    return 10 + (x**2 - 10 * np.cos(2*np.pi * x))

# find non-zero fitness for selection


def get_fitness(pred):
    return pred - np.max(pred)


# convert binary DNA to decimal and normalize it to a range(0, 5)
def translateDNA(pop):
    return pop.dot(2 ** np.arange(INDIV_SIZE)[::-1]) / float(2**INDIV_SIZE-1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]


def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE,
                           replace=True, p=fitness/fitness.sum())
    return pop[idx]


def crossover(parent, pop):     # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        # select another individual from pop
        i_ = np.random.randint(0, POP_SIZE, size=1)
        cross_points = np.random.randint(0, 2, size=INDIV_SIZE).astype(
            np.bool)   # choose crossover points
        # mating and produce one child
        parent[cross_points] = pop[i_, cross_points]
    return parent


def mutate(child):
    for point in range(INDIV_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


def main():

    # initialize the pop DNA
    pop = np.random.randint(2, size=(POP_SIZE, INDIV_SIZE))

    plt.ion()       # something about plotting
    x = np.linspace(*X_BOUND, 200)
    plt.plot(x, F(x))

    for i in range(N_GENERATIONS):
        # compute function value by extracting DNA
        F_values = F(translateDNA(pop))

        if 'sca' in locals():
            sca.remove()
        sca = plt.scatter(translateDNA(pop), F_values,
                          s=200, lw=0, c='red', alpha=0.5)
        plt.pause(0.05)

        # GA part (evolution)
        fitness = get_fitness(F_values)
        print("Fittest solution: ", pop[np.argmin(fitness), :])

        pop = select(pop, fitness)
        pop_copy = pop.copy()

        for parent in pop:
            child = crossover(parent, pop_copy)
            child = mutate(child)
            parent[:] = child  # parent is replaced by its child

    plt.ioff()
    plt.show()
    print('The best individual found is {0}'.format(pop[0]))
    print('The optimal solution is x={0} and f(x)={1}'.format(
        translateDNA(pop[0]), F(translateDNA(pop[0]))))

    arquivo = open("Genetic_Algorithm/rastrigin_GA.txt", "a")
    # arquivo.write("Algoritmo de Algoritmo Genetico com a funcao de Rastrigin\n\n")

    arquivo.write('The best individual found is {0}'.format(
        pop[0]) + '\nThe optimal solution is x={0} and f(x)={1} \n\n'.format(translateDNA(pop[0]), F(translateDNA(pop[0]))))


if __name__ == "__main__":
    main()
