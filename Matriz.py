# Not working!


import csv
import pandas as pd

tarefa_l = []
server_l = []

tarefa_s = []
server_s = []

with open('p23_l.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        server_l.append(row)
    for column in reader:
        tarefa_l.append(column)

# with open('p23_s.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         server_s.append(row)
#     for column in reader:
#         tarefa_s.append(column)

print("Linhas:", server_s)
print("Colunas:", tarefa_s)
