import numpy as np
import math
import statistics
import sys
import time
import matplotlib.pyplot as plt


arquivo = open("Entropia_Cruzada/entropia_ackley_valores.txt", "w")
arquivo.write("Algoritmo de Entropia Cruzada com a funcao de Ackley\n\n")
arquivo.write("Time" + ", " + "Iteracoes" + ", " + "x1" + ", " + "x2\n")
for i in range(100):
    # arquivo.write("Iteracao: " + str(i) + "\n")
    mu1 = 2
    mu2 = 2
    # 5 é a variancia
    sigma1 = np.sqrt(5)
    sigma2 = np.sqrt(5)
    t = 0
    maxits = 50
    N = 100
    Ne = 10
    epsilon = sys.float_info.epsilon

    def ackley(x1, x2):
        # x1 = x1 + 5
        # x2 = x2 - 3
        return 20 + np.exp(1) - 20 * np.exp((-0.2) * np.sqrt((0.5)*(x1**2 + x2**2)))

    resX = []
    resY1 = []
    resY2 = []

    # Ne nº de elite
    start_time = time.time_ns()

    # while maxits not exceeded and not converged
    while (t < maxits and sigma2 > epsilon):  # and sigma2 > epsilon
        # Obtain N samples from current sampling distribution
        x1 = np.random.normal(mu1, sigma2, N)
        x2 = np.random.normal(mu2, sigma2, N)

        # Evaluate onjective function at sample points
        S = ackley(x1, x2)

        #
        S, x1, x2 = zip(*sorted(zip(S, x1, x2), reverse=False))
        #
        mu1 = statistics.mean(x1[0:Ne])
        sigma1 = np.sqrt(np.var(x1[0:Ne]))

        mu2 = statistics.mean(x2[0:Ne])
        sigma2 = np.sqrt(np.var(x2[0:Ne]))
        t = t+1
        resX.append(t)
        resY1.append(mu1)
        resY2.append(mu2)

    # print("Time Total =", (time.time_ns() - start_time)/1e6, "ms")
    # arquivo.write("Time: " + str((time.time_ns() - start_time)/1e6) + "ms\n")
    # arquivo.write("N.Iteracoes \tSolucao\n")

    arquivo.write(str((time.time_ns() - start_time)/1e6) +
                  ", " + str(t) + ", " + str(mu1) + ", " + str(mu2) + "\n")
    # arquivo.write(str(t)+"\t\t\t\tmu2=  "+str(mu2)+"\n\n")
    # ("Time Total =" +(time.time_ns() - start_time)/1e6 + "ms\n\n")
    print(time)
    print("mu1 =", mu1)
    print("mu2 =", mu2)
plt.title('Entropia cruzada com a função Ackley')
plt.plot(resX, resY1, 'o-')
plt.plot(resX, resY2, 'o-')
plt.legend([' x1', ' x2'])
plt.show()
arquivo.close
