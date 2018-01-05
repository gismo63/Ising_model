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
	
		i = np.random.randint(0,rows)
		j = np.random.randint(0,columns)
	
		deltaE=2*J*matrix[i][j]*(matrix[i-1][j]+matrix[(i+1) % (rows)][j]+matrix[i][j-1]+matrix[i][(j+1) % (columns)])+2*h*matrix[i][j]

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
	for i in range(rows):
		for j in range(columns):
			tot_e += -1*(J*matrix[i][j]*(matrix[i-1][j]+matrix[i][j-1])+h*matrix[i][j])#each pair of sites should be counted only once
	return tot_e



start_time = time.time()

J=1.
columns=8
rows=8
h=0
T_c=2.2692
steps=2**16
averageing_steps = 2**8

T_array = np.random.normal(T_c, 0.5, 300)
T_array = T_array[(T_array>1.5) & (T_array<3.5)]
num_temps = len(T_array)

energy = np.zeros(num_temps)
magnetization = np.zeros(num_temps)
specheat = np.zeros(num_temps)
magsuscep = np.zeros(num_temps)



initial_matrix = np.random.choice([-1,1],size=(rows,columns))


for i in range(num_temps):
	neg_beta = -1./T_array[i]
	E = np.zeros(averageing_steps)
	M = np.zeros(averageing_steps)
	f_matrix = metrop(initial_matrix, steps, neg_beta)
	
	for j in range(averageing_steps):
		f_matrix = metrop(f_matrix, rows*columns, neg_beta)
		E[j] = tot_energy(f_matrix)
		M[j] = mag(f_matrix)
	energy[i] = np.sum(E)
	magnetization[i] = abs(np.sum(M))
	specheat[i] = np.sum(E*E) - np.sum(E)*np.sum(E)/averageing_steps
	magsuscep[i] = np.sum(M*M) - np.sum(M)*np.sum(M)/averageing_steps

c = rows*columns*averageing_steps

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
