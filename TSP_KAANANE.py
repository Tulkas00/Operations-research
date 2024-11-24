#!/usr/bin/env python
# coding: utf-8

# In[6]:


from random import *
import numpy as np
import math
import matplotlib.pyplot as plt
import tsplib95
import pandas as pd
from itertools import permutations
import copy
import matplotlib.pyplot as plt
import seaborn as sns


# In[41]:


infile = open(r'C:\Users\KAANANE Youssef\Desktop\tsplib-master\eil101.tsp', 'r')

Name = infile.readline().strip().split()[2] # NAME
FileType = infile.readline().strip().split()[2] # TYPE
Comment = infile.readline().strip().split()[2] # COMMENT
Dimension = infile.readline().strip().split()[2] # DIMENSION
EdgeWeightType = infile.readline().strip().split()[2] # EDGE_WEIGHT_TYPE
infile.readline()

nodelist = []
N = int(Dimension)
for i in range(0, N):
    Id,x,y = infile.readline().strip().split()
    nodelist.append([int(Id),float(x), float(y)])

infile.close()


# In[42]:


def dist2cities(city1,city2) :
    return math.sqrt((city1[1]-city2[1])**2+(city1[2]-city2[2])**2)
n=len(nodelist)
dist = np.zeros((n,n))
for i in range(n) :
    for j in range(i+1,n) :
        dist[i,j] = dist2cities(nodelist[i],nodelist[j])
def get_distance(i,j):
    if i<j :
        return dist[i,j]
    else :
        return dist[j,i]
    
dist = dist + np.transpose(dist)
df_dist = pd.DataFrame(dist,columns = [i for i in range(n)])
df_dist.replace({0.0 : np.inf},inplace=True)

def next_city(Id,revized = list(),total_path = 0) :
    Sr = df_dist[Id].copy()
    Id_next = Sr[Sr == min(Sr)].index[0]
    while Id_next in revized :
        Sr[Id_next] = np.inf
        Id_next = Sr[Sr == min(Sr)].index[0]
    total_path+=min(Sr)
    return Id_next , revized + [Id_next] , total_path

def get_tour(Id) :
    revized = [Id]
    Id_next = Id
    total_path = 0
    while len(revized) < n :
        Id_next,revized , total_path = next_city(Id_next,revized,total_path)
        
    total_path+=get_distance(Id,revized[-1])
    revized+=[Id]
    return revized , total_path

Tours = dict()
Tours["Ville de depart"] = list()
Tours["Tour"] = list()
Tours["Distance"] = list()

for i in range(n) :
    x,y = get_tour(i)
    Tours["Ville de depart"].append(i)
    Tours["Tour"].append(x)
    Tours["Distance"].append(y)
    
df_tour = pd.DataFrame.from_dict(Tours)
id_Min=df_tour["Distance"].idxmin()
Tour_Optimal= df_tour["Tour"][id_Min]
print("la langueur du tour optimal est :", min(df_tour["Distance"]) )

Tour_coord=[]
for i in range(len(Tour_Optimal)):
    
    for j in range (len(nodelist)):
        if nodelist[j][0]==Tour_Optimal[i]:
            k=j
            Tour_coord.append([Tour_Optimal[i],nodelist[k][1], nodelist[k][2]])

X = list(map(lambda x:x[1] , Tour_coord))
Y = list(map(lambda x:x[2] , Tour_coord))
Z = list(map(lambda x:x[0] , Tour_coord))
plt.figure(figsize=(20,10))
for i in range(n):
    x=X[i]
    y=Y[i]
    z=Z[i]

    plt.plot(x,y)
    an2 = plt.annotate(z, xy=(x,y))
    
plt.plot(X,Y)
plt.show()


# In[43]:


Tour_coord
 
    


# In[38]:


infile = open(r'C:\Users\KAANANE Youssef\Desktop\tsplib-master\berlin52.tsp', 'r')

Name = infile.readline().strip().split()[2] # NAME
FileType = infile.readline().strip().split()[2] # TYPE
Comment = infile.readline().strip().split()[2] # COMMENT
Dimension = infile.readline().strip().split()[2] # DIMENSION
EdgeWeightType = infile.readline().strip().split()[2] # EDGE_WEIGHT_TYPE
infile.readline()

nodelist = []
N = int(Dimension)
for i in range(0, N):
    Id,x,y = infile.readline().strip().split()
    nodelist.append([int(Id),float(x), float(y)])

infile.close()


# In[46]:


infile = open(r'C:\Users\KAANANE Youssef\Desktop\tsplib-master\berlin52.tsp', 'r')

Name = infile.readline().strip().split()[1] # NAME
FileType = infile.readline().strip().split()[1] # TYPE
Comment = infile.readline().strip().split()[1] # COMMENT
Dimension = infile.readline().strip().split()[1] # DIMENSION
EdgeWeightType = infile.readline().strip().split()[1] # EDGE_WEIGHT_TYPE
infile.readline()

nodelist = []
N = int(Dimension)
for i in range(0, N):
    Id,x,y = infile.readline().strip().split()
    nodelist.append([int(Id),float(x), float(y)])

infile.close()


# In[48]:


def dist2cities(city1,city2) :
    return math.sqrt((city1[1]-city2[1])**2+(city1[2]-city2[2])**2)
n=len(nodelist)
dist = np.zeros((n,n))
for i in range(n) :
    for j in range(i+1,n) :
        dist[i,j] = dist2cities(nodelist[i],nodelist[j])
def get_distance(i,j):
    if i<j :
        return dist[i,j]
    else :
        return dist[j,i]
    
dist = dist + np.transpose(dist)
df_dist = pd.DataFrame(dist,columns = [i for i in range(n)])
df_dist.replace({0.0 : np.inf},inplace=True)

def next_city(Id,revized = list(),total_path = 0) :
    Sr = df_dist[Id].copy()
    Id_next = Sr[Sr == min(Sr)].index[0]
    while Id_next in revized :
        Sr[Id_next] = np.inf
        Id_next = Sr[Sr == min(Sr)].index[0]

    total_path+=min(Sr)
    return Id_next , revized + [Id_next] , total_path

def get_tour(Id) :
    revized = [Id]
    Id_next = Id
    total_path = 0
    while len(revized) < n :
        Id_next,revized , total_path = next_city(Id_next,revized,total_path)
        
    total_path+=get_distance(Id,revized[-1])
    revized+=[Id]
    return revized , total_path

Tours = dict()
Tours["Ville de depart"] = list()
Tours["Tour"] = list()
Tours["Distance"] = list()

for i in range(n) :
    x,y = get_tour(i)
    Tours["Ville de depart"].append(i)
    Tours["Tour"].append(x)
    Tours["Distance"].append(y)
    
df_tour = pd.DataFrame.from_dict(Tours)
id_Min=df_tour["Distance"].idxmin()
Tour_Optimal= df_tour["Tour"][id_Min]
print("la langueur du tour optimal est :", min(df_tour["Distance"]) )

Tour_coord=[]
for i in range(len(Tour_Optimal)):
    
    for j in range (len(nodelist)):
        if nodelist[j][0]==Tour_Optimal[i]:
            k=j
            Tour_coord.append([Tour_Optimal[i],nodelist[k][1], nodelist[k][2]])

X = list(map(lambda x:x[1] , Tour_coord))
Y = list(map(lambda x:x[2] , Tour_coord))
Z = list(map(lambda x:x[0] , Tour_coord))

plt.figure(figsize=(20,10))
for i in range(n):
    x=X[i]
    y=Y[i]
    z=Z[i]

    plt.plot(x,y)
    an2 = plt.annotate(z, xy=(x,y))
    
plt.plot(X,Y,color='green')
plt.show()


# In[40]:


Tour_coord


# In[49]:


def Distance(x1, y1, x2, y2):
    return sqrt((int(x2) - int(x1)) ** 2 + (int(y2) - int(y1)) ** 2)


# In[50]:


def Donnéesvilles(file_directrory):
    données = tsplib95.load(file_directrory)
    nombre_villes = max(list(données.get_nodes()))
    matrice_distance = np.empty((nombre_villes, nombre_villes))
    for i in range(nombre_villes):
        for j in range(nombre_villes):
            if i == j:
                matrice_distance[i][i] = 1
            elif i < j:      
                matrice_distance[i][j] = matrice_distance[j][i] = Distance(données.node_coords[i+1][0],données.node_coords[i+1][1],données.node_coords[j+1][0], données.node_coords[j+1][1])
    return matrice_distance


# In[51]:


def coordonnées_ville(file_directrory):
    données = tsplib95.load(file_directrory)
    nombre_villes = max(list(données.get_nodes()))
    coordonnées_ville = []
    for i in range(nombre_villes):
        coordonnées_ville.append([données.node_coords[i+1][0], données.node_coords[i+1][1]])
    return coordonnées_ville


# In[52]:


def closest_city(city, previous_cities, distance_matrix):
    for elm in previous_cities :
        distance_matrix[city][elm] = np.max(distance_matrix[city])+1
    return np.argmin(distance_matrix[city])


# In[53]:


def greedy_nearestneighbor(indice,distance_matrix):
    cities = [i for i in range(len(distance_matrix))]
    solution = [indice]
    cities.remove(solution[0])
    while cities:
        distance_matrix_2 = copy.deepcopy(distance_matrix)
        next_city = closest_city(solution[-1:][0],solution, distance_matrix_2)
        solution.append(next_city)
        cities.remove(next_city)
    solution += [solution[0]]
    return solution


# In[54]:


def Fitness(solution, matrix_distance):
    fitness=0
    for i in range(len(solution)-1):
            fitness=fitness+matrix_distance[solution[i]][solution[i+1]]
    return fitness


# In[55]:


def tracer_solution(solution,coordonées_ville):
    X=[]
    Y=[]
    for sol in solution:
       X.append(coordonées_ville[sol][0])
       Y.append(coordonées_ville[sol][1])
    return [X,Y]


# In[57]:


# I.2 Recherche locale  (decente) : #


# In[58]:


def city_swap(solution,distance_matrix):
    liste = copy.deepcopy(solution)
    liste.pop(len(liste) - 1)
    tot_liste =[]
    tot_liste.append([solution, Fitness(solution,distance_matrix)])
    for i in range(len(solution)-1):
        for j in range(i+1, len(solution)-1):
            new_liste = liste.copy()
            new_liste[i], new_liste[j] = new_liste[j], new_liste[i]
            new_liste += [new_liste[0]]
            tot_liste.append([new_liste,Fitness(new_liste,distance_matrix)])
    return tot_liste


# In[59]:


def LS_First_improvement(solution,matrix_distance):
    X=solution
    amélioration=True
    while amélioration==True:
        amélioration=False
        X1=X
        voisinage = city_swap(X1, matrix_distance)
        for i in range(len(voisinage)):
            Y=voisinage[i][1]
            if Y-Fitness(X1,matrix_distance)<0:
                X=voisinage[i][0]
                amélioration= True
                break
    return X


# In[60]:


def LS_best_impro(solution,matrix_distance):
    X=solution
    amélioration = True
    while amélioration == True:
        amélioration=False
        X1=X
        voisinage = city_swap(X1, matrix_distance)
        for i in range(len(voisinage)):
            Y = voisinage[i][1]
            d=Fitness(X1,matrix_distance)
            if Y-d <0:
                X= voisinage[i][0]
                amélioration = True
    return X1


# In[61]:


def LS_twoopt(solution, matrix_distance):
    min_change=0
    C=solution
    for i in range(len(C)-2):
        for j in range(i+2,len(C)-1):
            actual_cost = matrix_distance[C[i]][C[i+1]]+ matrix_distance[C[j]][C[j+1]]
            new_cost = matrix_distance[C[i]][C[j]] + matrix_distance[C[i+1]][C[j+1]]
            change=new_cost-actual_cost
            if change <min_change:
                min_change=change
                min_i=i
                min_j=j
    if min_change<0:
        C[min_i+1:min_j+1]=C[min_i+1:min_j+1][:-1]
    return C


# In[63]:


def main():
    matrix_distance = Donnéesvilles(r'C:\Users\KAANANE Youssef\Desktop\tsplib-master\berlin52.tsp')
    a =greedy_nearestneighbor(30,matrix_distance)
    fit1=Fitness(a,matrix_distance)
    print(a)
    print(fit1)

    solution2= LS_twoopt(solution, matrix_distance)
    fit2=Fitness(solution2,matrix_distance)
    print(solution2)
    print(fit2)
    Coord=coordonnées_ville(r'C:\Users\saman\Desktop\Cours 3A\Option GI\Recherche Opérationnelle\berlin52.tsp')
    Z=tracer_solution(solution,Coord)
    plt.plot(Z[0],Z[1])
    plt.show()
    H=city_swap(solution,matrix_distance)
    voisinage= pd.DataFrame(H, columns=['liste', 'qualite'])
    print(voisinage)
    first_improv=LS_First_improvement(solution, matrix_distance)
    print(first_improv)
    best=LS_best_impro(solution,matrix_distance)
    print(best)
    SA=simulated_annealing(solution,400,20,0.9,200,matrix_distance)
    print(SA)


main()


# In[64]:


def main():
    matrix_distance = Donnéesvilles(r'C:\Users\KAANANE Youssef\Desktop\tsplib-master\eil101.tsp')
    a =greedy_nearestneighbor(30,matrix_distance)
    fit1=Fitness(a,matrix_distance)
    print(a)
    print(fit1)




main()


# In[67]:


def main():
    matrix_distance = Donnéesvilles(r'C:\Users\KAANANE Youssef\Desktop\tsplib-master\eil101.tsp')
    a =greedy_nearestneighbor(34, matrix_distance)
    fit1=Fitness(a,matrix_distance)
    first = LS_First_improvement(a, matrix_distance)
    best = LS_best_impro(a, matrix_distance)
    two_opt = LS_twoopt(a, matrix_distance)

    print(first)
    print(Fitness(first, matrix_distance))
    

main()


# In[68]:


def main():
    matrix_distance = Donnéesvilles(r'C:\Users\KAANANE Youssef\Desktop\tsplib-master\eil101.tsp')
    a =greedy_nearestneighbor(34, matrix_distance)
    fit1=Fitness(a,matrix_distance)
    first = LS_First_improvement(a, matrix_distance)
    best = LS_best_impro(a, matrix_distance)
    two_opt = LS_twoopt(a, matrix_distance)

    print(best)
    print(Fitness(best, matrix_distance))
    

main()


# In[69]:


def main():
    matrix_distance = Donnéesvilles(r'C:\Users\KAANANE Youssef\Desktop\tsplib-master\eil101.tsp')
    a =greedy_nearestneighbor(34, matrix_distance)
    fit1=Fitness(a,matrix_distance)
    first = LS_First_improvement(a, matrix_distance)
    best = LS_best_impro(a, matrix_distance)
    two_opt = LS_twoopt(a, matrix_distance)
    
    print(two_opt)
    print(Fitness(two_opt, matrix_distance))
    

main()


# In[73]:


def main():
    matrix_distance = Donnéesvilles(r'C:\Users\KAANANE Youssef\Desktop\tsplib-master\berlin52.tsp')
    a =greedy_nearestneighbor(30,matrix_distance)
    fit1=Fitness(a,matrix_distance)
    print(a)
    print(fit1)

    solution2= LS_twoopt(a, matrix_distance)
    fit2=Fitness(a,matrix_distance)
    print(solution2)
    print(fit2)
    Coord=coordonnées_ville(r'C:\Users\KAANANE Youssef\Desktop\tsplib-master\berlin52.tsp')
    Z=tracer_solution(a,Coord)
    plt.plot(Z[0],Z[1])
    plt.show()
    H=city_swap(a,matrix_distance)
    voisinage= pd.DataFrame(H, columns=['liste', 'qualite'])
    print(voisinage)
    first_improv=LS_First_improvement(a, matrix_distance)
    print(first_improv)
    best=LS_best_impro(a,matrix_distance)
    print(best)
    SA=simulated_annealing(a,400,20,0.9,200,matrix_distance)
    print(SA)


main()


# In[74]:


def main():
    matrix_distance = Donnéesvilles(r'C:\Users\KAANANE Youssef\Desktop\tsplib-master\eil101.tsp')
    a =greedy_nearestneighbor(30,matrix_distance)
    fit1=Fitness(a,matrix_distance)
    print(a)
    print(fit1)

    solution2= LS_twoopt(a, matrix_distance)
    fit2=Fitness(a,matrix_distance)
    print(solution2)
    print(fit2)
    Coord=coordonnées_ville(r'C:\Users\KAANANE Youssef\Desktop\tsplib-master\eil101.tsp')
    Z=tracer_solution(a,Coord)
    plt.plot(Z[0],Z[1])
    plt.show()
    H=city_swap(a,matrix_distance)
    voisinage= pd.DataFrame(H, columns=['liste', 'qualite'])
    print(voisinage)
    first_improv=LS_First_improvement(a, matrix_distance)
    print(first_improv)
    best=LS_best_impro(a,matrix_distance)
    print(best)
    #SA=simulated_annealing(a,400,20,0.9,200,matrix_distance)
    #print(SA)


main()


# In[ ]:




