#!/usr/bin/env python

"""
This program calculates the magnetization, energy, heat capacity and magnetic susceptibility
per site of the 3d fcc lattice at various temperatures around the critical temperature
and plots graphs of them
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import time

"""
There isn't a particularly nice way to arrange the fcc lattice as there was with the bcc lattice,
it is easier to consider the cpp structure which is just the fcc rotated. This consists of layers
of 3 triangular lattices stacked on top of each other
The three matrices in the program represent the three different triangular lattices
Since matricies can't properly represent the triangular matricies the nearest neighbours are 
slightly different for each layer. In order to figure these out I drew the layers out on paper
each spin has 12 nearest neighbours
"""


#See analysis.py for explanation of most of the code
def metrop(matrix1, matrix2, matrix3, iterations, neg_beta):
    m=0
    while m < iterations:
        v = np.random.choice([0,1,2])#choose randomly between the 3 layers
        if v==0:
            i = np.random.randint(0,N)
            j = np.random.randint(0,N)
            k = np.random.randint(0,N/3)
            
            if j%2:
                deltaE=2*(J*matrix1[i][j][k]*(matrix1[i-1][j][k]+matrix1[(i+1)%N][j][k]+matrix1[(i+1)%N][j-1][k]+matrix1[i][j-1][k]+matrix1[i][(j+1)%N][k]+matrix1[(i+1)%N][(j+1)%N][k] + matrix2[i][j-1][k]+matrix2[(i+1)%N][j-1][k]+matrix2[i][j][k] + matrix3[i][j-1][k-1]+matrix3[i][j][k-1]+matrix3[(i+1)%N][j][k-1])+h*matrix1[i][j][k])
            else:
                deltaE=2*(J*matrix1[i][j][k]*(matrix1[i-1][j][k]+matrix1[(i+1)%N][j][k]+matrix1[i-1][j-1][k]+matrix1[i][j-1][k]+matrix1[i][(j+1)%N][k]+matrix1[i-1][(j+1)%N][k] + matrix2[i][j-1][k]+matrix2[i-1][j-1][k]+matrix2[i][j][k] + matrix3[i][j-1][k-1]+matrix3[i][j][k-1]+matrix3[i-1][j][k-1])+h*matrix1[i][j][k])
            
            if deltaE<=0:
                matrix1[i][j][k] *= -1
            elif random.random() < np.exp(deltaE*neg_beta):
                matrix1[i][j][k] *= -1
            m+=1
        elif v==1:
            i = np.random.randint(0,N)
            j = np.random.randint(0,N)
            k = np.random.randint(0,N/3)
            
            if j%2:
                deltaE=2*(J*matrix2[i][j][k]*(matrix2[i-1][j][k]+matrix2[(i+1)%N][j][k]+matrix2[(i+1)%N][j-1][k]+matrix2[i][j-1][k]+matrix2[i][(j+1)%N][k]+matrix2[(i+1)%N][(j+1)%N][k] + matrix3[i][j][k]+matrix3[(i+1)%N][j][k]+matrix3[i][(j+1)%N][k] + matrix1[i][(j+1)%N][k]+matrix1[(i+1)%N][(j+1)%N][k]+matrix1[i][j][k])+h*matrix1[i][j][k])
            else:
                deltaE=2*(J*matrix2[i][j][k]*(matrix2[i-1][j][k]+matrix2[(i+1)%N][j][k]+matrix2[i-1][j-1][k]+matrix2[i][j-1][k]+matrix2[i][(j+1)%N][k]+matrix2[i-1][(j+1)%N][k] + matrix3[i][j][k]+matrix3[i-1][j][k]+matrix3[i][(j+1)%N][k] + matrix1[i][(j+1)%N][k]+matrix1[i-1][(j+1)%N][k]+matrix1[i][j][k])+h*matrix1[i][j][k])
            if deltaE<=0:
                matrix2[i][j][k] *= -1
            elif random.random() < np.exp(deltaE*neg_beta):
                matrix2[i][j][k] *= -1
            m+=1
        else:
            i = np.random.randint(0,N)
            j = np.random.randint(0,N)
            k = np.random.randint(0,N/3)	
            
            if j%2:
                deltaE=2*(J*matrix3[i][j][k]*(matrix3[i-1][j][k]+matrix3[(i+1)%N][j][k]+matrix3[(i+1)%N][j-1][k]+matrix3[i][j-1][k]+matrix3[i][(j+1)%N][k]+matrix3[(i+1)%N][(j+1)%N][k] + matrix1[i-1][j][(k+1)%(N/3)]+matrix1[i][j][(k+1)%(N/3)]+matrix1[i][(j+1)%N][(k+1)%(N/3)] + matrix2[i-1][j][k]+matrix2[i][j][k]+matrix2[i][j-1][k])+h*matrix1[i][j][k])
            else:
                deltaE=2*(J*matrix3[i][j][k]*(matrix3[i-1][j][k]+matrix3[(i+1)%N][j][k]+matrix3[i-1][j-1][k]+matrix3[i][j-1][k]+matrix3[i][(j+1)%N][k]+matrix3[i-1][(j+1)%N][k] + matrix1[(i+1)%N][j][(k+1)%(N/3)]+matrix1[i][j][(k+1)%(N/3)]+matrix1[i][(j+1)%N][(k+1)%(N/3)] + matrix2[(i+1)%N][j][k]+matrix2[i][j][k]+matrix2[i][j-1][k])+h*matrix1[i][j][k])
            if deltaE<=0:
                matrix3[i][j][k] *= -1
            elif random.random() < np.exp(deltaE*neg_beta):
                matrix3[i][j][k] *= -1
            m+=1

	
    return matrix1, matrix2, matrix3


def mag(matrix1, matrix2, matrix3):
    return (np.sum(matrix1)+ np.sum(matrix2) + np.sum(matrix3))

def tot_energy(matrix1, matrix2, matrix3):
    tot_e = 0
    arr = range(N)
    for i in arr:
        for j in arr:
            for k in range(N/3):
                if j%2:
                    tot_e+=-1*(J*matrix1[i][j][k]*(matrix1[i-1][j][k]+matrix1[(i+1)%N][j][k]+matrix1[(i+1)%N][j-1][k]+matrix1[i][j-1][k]+matrix1[i][(j+1)%N][k]+matrix1[(i+1)%N][(j+1)%N][k] + matrix2[i][j-1][k]+matrix2[(i+1)%N][j-1][k]+matrix2[i][j][k] + matrix3[i][j-1][k-1]+matrix3[i][j][k-1]+matrix3[(i+1)%N][j][k-1])+2*h*matrix1[i][j][k])
                else:
                    tot_e+=-1*(J*matrix1[i][j][k]*(matrix1[i-1][j][k]+matrix1[(i+1)%N][j][k]+matrix1[i-1][j-1][k]+matrix1[i][j-1][k]+matrix1[i][(j+1)%N][k]+matrix1[i-1][(j+1)%N][k] + matrix2[i][j-1][k]+matrix2[i-1][j-1][k]+matrix2[i][j][k] + matrix3[i][j-1][k-1]+matrix3[i][j][k-1]+matrix3[i-1][j][k-1])+2*h*matrix1[i][j][k])
    for i in arr:
        for j in arr:
            for k in range(N/3):
                if j%2:
                    tot_e+=-1*(J*matrix2[i][j][k]*(matrix2[i-1][j][k]+matrix2[(i+1)%N][j][k]+matrix2[(i+1)%N][j-1][k]+matrix2[i][j-1][k]+matrix2[i][(j+1)%N][k]+matrix2[(i+1)%N][(j+1)%N][k] + matrix3[i][j][k]+matrix3[(i+1)%N][j][k]+matrix3[i][(j+1)%N][k] + matrix1[i][(j+1)%N][k]+matrix1[(i+1)%N][(j+1)%N][k]+matrix1[i][j][k])+2*h*matrix1[i][j][k])
                else:
                    tot_e+=-1*(J*matrix2[i][j][k]*(matrix2[i-1][j][k]+matrix2[(i+1)%N][j][k]+matrix2[i-1][j-1][k]+matrix2[i][j-1][k]+matrix2[i][(j+1)%N][k]+matrix2[i-1][(j+1)%N][k] + matrix3[i][j][k]+matrix3[i-1][j][k]+matrix3[i][(j+1)%N][k] + matrix1[i][(j+1)%N][k]+matrix1[i-1][(j+1)%N][k]+matrix1[i][j][k])+2*h*matrix1[i][j][k])
    for i in arr:
        for j in arr:
            for k in range(N/3):
                if j%2:
                    tot_e+=-1*(J*matrix3[i][j][k]*(matrix3[i-1][j][k]+matrix3[(i+1)%N][j][k]+matrix3[(i+1)%N][j-1][k]+matrix3[i][j-1][k]+matrix3[i][(j+1)%N][k]+matrix3[(i+1)%N][(j+1)%N][k] + matrix1[i-1][j][(k+1)%(N/3)]+matrix1[i][j][(k+1)%(N/3)]+matrix1[i][(j+1)%N][(k+1)%(N/3)] + matrix2[i-1][j][k]+matrix2[i][j][k]+matrix2[i][j-1][k])+2*h*matrix1[i][j][k])
                else:
                    tot_e+=-1*(J*matrix3[i][j][k]*(matrix3[i-1][j][k]+matrix3[(i+1)%N][j][k]+matrix3[i-1][j-1][k]+matrix3[i][j-1][k]+matrix3[i][(j+1)%N][k]+matrix3[i-1][(j+1)%N][k] + matrix1[(i+1)%N][j][(k+1)%(N/3)]+matrix1[i][j][(k+1)%(N/3)]+matrix1[i][(j+1)%N][(k+1)%(N/3)] + matrix2[(i+1)%N][j][k]+matrix2[i][j][k]+matrix2[i][j-1][k])+2*h*matrix1[i][j][k])

    return tot_e/2



start_time = time.time()

J=1.
N=9
h=0
T_c=9.
steps=2**22
averageing_steps = 2**10

T_array = np.random.normal(T_c, 0.7, 300)
T_array = T_array[(T_array>7) & (T_array<12)]
num_temps = len(T_array)

energy = np.zeros(num_temps)
magnetization = np.zeros(num_temps)
specheat = np.zeros(num_temps)
magsuscep = np.zeros(num_temps)



initial_matrix1 = np.random.choice([-1,1],size=(N,N,N/3))
initial_matrix2 = np.random.choice([-1,1],size=(N,N,N/3))                        
initial_matrix3 = np.random.choice([-1,1],size=(N,N,N/3))                        

for i in range(num_temps):
    neg_beta = -1./T_array[i]
    E = np.zeros(averageing_steps)
    M = np.zeros(averageing_steps)
    f_matrix1, f_matrix2, f_matrix3 = metrop(initial_matrix1, initial_matrix2, initial_matrix3, steps, neg_beta)
	
    for j in range(averageing_steps):
        f_matrix1, f_matrix2, f_matrix3 = metrop(f_matrix1, f_matrix2, f_matrix3, N**4, neg_beta)
        E[j] = tot_energy(f_matrix1,f_matrix2, f_matrix3)
        M[j] = mag(f_matrix1,f_matrix2, f_matrix3)
    energy[i] = np.sum(E)
    magnetization[i] = abs(np.sum(M))
    specheat[i] = np.sum(E*E) - np.sum(E)*np.sum(E)/averageing_steps
    magsuscep[i] = np.sum(M*M) - np.sum(M)*np.sum(M)/averageing_steps

c = N**3*averageing_steps

energy = energy / (c)
magnetization = magnetization / (c)
specheat = (specheat / (T_array**2)) / (c)
magsuscep = (magsuscep / (T_array)) / (c)
	
plt.plot(T_array, magnetization, 'o')

plt.figure()

plt.plot(T_array, energy, 'o')

plt.figure()

plt.plot(T_array, specheat, 'o')

plt.figure()

plt.plot(T_array, magsuscep, 'o')


print f_matrix1

print "%s seconds" % (time.time() - start_time)
plt.show()
