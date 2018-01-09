#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from matplotlib import animation



#im = plt.imshow(initial_matrix,cmap='Greys', animated=True)


#spin_change=0
#equilib = []
#img = []




def metrop(matrix, iterations, neg_beta):
    k=0
    while k < iterations:
        i = np.random.randint(0,N)
        j = np.random.randint(0,N)
        if i%2:
            deltaE=2*(J*matrix[i][j]*(matrix[i][j-1]+matrix[i][(j+1)%N]+matrix[i-1][(j+1)%N]+matrix[i-1][j]+matrix[(i+1)%N][(j+1)%N]+matrix[(i+1)%N][j])+h*matrix[i][j])
        else:
            deltaE=2*(J*matrix[i][j]*(matrix[i][j-1]+matrix[i][(j+1)%N]+matrix[i-1][j-1]+matrix[i-1][j]+matrix[(i+1)%N][j-1]+matrix[(i+1)%N][j])+h*matrix[i][j])
        if deltaE<=0:
            matrix[i][j] *= -1
            #spin_change += 1
        elif random.random() < np.exp(deltaE*neg_beta):
            matrix[i][j] *= -1
            #spin_change += 1
        #equilib.append(spin_change)
        k+=1
        #if k%10000 == 0:
            #img.append([plt.imshow(matrix,cmap='Greys')])
    return matrix


def mag(matrix):
    return np.sum(matrix)

def tot_energy(matrix):
    tot_e = 0
    for i in range(N):
        for j in range(N):
            if i%2:
                tot_e+=-1*(J*matrix[i][j]*(matrix[i][j-1]+matrix[i][(j+1)%N]+matrix[i-1][(j+1)%N]+matrix[i-1][j]+matrix[(i+1)%N][(j+1)%N]+matrix[(i+1)%N][j])+2*h*matrix[i][j])
            else:
                tot_e+=-1*(J*matrix[i][j]*(matrix[i][j-1]+matrix[i][(j+1)%N]+matrix[i-1][j-1]+matrix[i-1][j]+matrix[(i+1)%N][j-1]+matrix[(i+1)%N][j])+2*h*matrix[i][j])
    return tot_e/2



start_time = time.time()

J=1.
N=8
h=0
T_c=3.5
steps=2**18
averageing_steps = 2**10

T_array = np.random.normal(T_c, 0.5, 30)
T_array = T_array[(T_array>2.5) & (T_array<4.5)]
num_temps = len(T_array)

energy = np.zeros(num_temps)
magnetization = np.zeros(num_temps)
specheat = np.zeros(num_temps)
magsuscep = np.zeros(num_temps)



initial_matrix = np.random.choice([-1,1],size=(N,N))


for i in range(num_temps):
    neg_beta = -1./T_array[i]
    E = np.zeros(averageing_steps)
    M = np.zeros(averageing_steps)
    f_matrix = metrop(initial_matrix, steps, neg_beta)
	
    for j in range(averageing_steps):
        f_matrix = metrop(f_matrix, 3*N*N, neg_beta)
        E[j] = tot_energy(f_matrix)
        M[j] = mag(f_matrix)
    energy[i] = np.sum(E)
    magnetization[i] = abs(np.sum(M))
    specheat[i] = np.sum(E*E) - np.sum(E)*np.sum(E)/averageing_steps
    magsuscep[i] = np.sum(M*M) - np.sum(M)*np.sum(M)/averageing_steps

c = N*N*averageing_steps

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

#equilib = equilib[::iterations/10000]



#fig=plt.figure()

#ani = animation.ArtistAnimation(fig, img, interval = 0, blit = True, repeat_delay = 1000)


#plt.figure()


#plt.imshow(f_matrix,cmap='Greys')




#plt.plot(equilib)

print f_matrix

print "%s seconds" % (time.time() - start_time)
plt.show()
