# a = [1, 2, 3, 4, 5]
# b = [3, 1, 6, 4, 7]

# b, a= zip(*sorted(zip(b, a), reverse=False))

# print(a)
# print(b)

# 20 + ((x1**2 - 10 * np.cos(2*np.pi *x)) + (x2**2 - 10 * np.cos(2*np.pi *x2)))


arquivo = open("arquivo.txt", "w")
arquivo.write("N.Iteracoes \tSolucao\n")
num = 20
for i in range(100):
    num = num+2
    arquivo.write(str(i+1)+"\t\t\t\t"+str(num)+"\n")
arquivo.close
