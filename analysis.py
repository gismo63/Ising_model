#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from matplotlib import animation




start_time = time.time()

J=1.
columns=12
rows=12

dt=0.1
T_array=np.arange(dt,4,dt)
h=0

iterations=1000000

isingmat = np.random.choice([-1,1],size=(rows,columns))

#im = plt.imshow(isingmat,cmap='Greys', animated=True)


#spin_change=0
#equilib = []
#img = []




def metrop(isingmat, iterations, T):
	k=0
	while k < iterations:
	
		i = np.random.randint(0,rows)
		j = np.random.randint(0,columns)
	
		deltaE=2*J*isingmat[i][j]*(isingmat[i-1][j]+isingmat[(i+1) % (rows)][j]+isingmat[i][j-1]+isingmat[i][(j+1) % (columns)])+2*h*isingmat[i][j]

		if deltaE<=0:
			isingmat[i][j] *= -1
			#spin_change += 1
		elif random.random() < np.exp(-deltaE/T):
			isingmat[i][j] *= -1
			#spin_change += 1
		#equilib.append(spin_change)
		k+=1
		#if k%10000 == 0:
			#img.append([plt.imshow(isingmat,cmap='Greys')])
	return isingmat

def sweep(matrix, iterations, T):
	deltaE = np.zeros((rows,columns))
	for k in range(iterations):
		for i in range(rows):
			for j in range(columns):
				deltaE[i][j]=-2*J*matrix[i][j]*(matrix[i-1][j]+matrix[(i+1) % (rows)][j]+matrix[i][j-1]+matrix[i][(j+1) % (columns)])-2*h*matrix[i][j]

		p_flip = np.exp((-deltaE)/(T))

		for i in range(rows):
			for j in range(columns):
				if p_flip[i][j]>random.random():
					matrix[i][j]*=-1
	return matrix

def mag(matrix):
	return abs(np.mean(matrix))


magnetization = []

for t in T_array:
	f_matrix = metrop(isingmat, iterations, t)
	magnetization.append(mag(f_matrix))
	
plt.plot(T_array, magnetization, 'o')
#equilib = equilib[::iterations/10000]



#fig=plt.figure()

#ani = animation.ArtistAnimation(fig, img, interval = 0, blit = True, repeat_delay = 1000)


#plt.figure()


#plt.imshow(f_matrix,cmap='Greys')




#plt.plot(equilib)

print f_matrix

print "%s seconds" % (time.time() - start_time)
plt.show()
