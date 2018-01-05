#!/usr/bin/env python
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import time

start_time = time.time()

J=1.
rows=50
T=1.
h=0
iterations=2**16

isingmat = np.random.choice([-1,1],size=rows)



k=0

deltaE=0
i=0
j=0
spin_change=0
equilib = []
neg_beta = -1./T



while k < iterations:
	
	i = np.random.randint(0,rows)
	
	deltaE=2*J*isingmat[i]*(isingmat[i-1]+isingmat[(i+1) % rows])+2*h*isingmat[i]

	if deltaE<=0:
		isingmat[i] *= -1
		spin_change += 1
	elif random.random() < np.exp(deltaE*neg_beta):
		isingmat[i] *= -1
		spin_change += 1
	equilib.append(spin_change)
	k+=1

equilib = equilib[::iterations/10000]




#plt.plot(np.exp(np.array(equilib)*1e-5))


print isingmat



#plt.imshow(isingmat,cmap='Greys')

#plt.figure()


plt.plot(equilib)
print "%s seconds" % (time.time() - start_time)
print np.mean(isingmat)
plt.show()



"""


#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from matplotlib import animation





#spin_change=0
#equilib = []




def metrop(matrix, iterations, neg_beta):
	k=0
	while k < iterations:
	
		i = np.random.randint(0,rows)
	
		deltaE=2*J*matrix[i]*(matrix[i-1]+matrix[(i+1) % rows])+2*h*matrix[i]

		if deltaE<=0:
			matrix[i] *= -1
			#spin_change += 1
		elif random.random() < np.exp(deltaE*neg_beta):
			matrix[i] *= -1
			#spin_change += 1
		#equilib.append(spin_change)
		k+=1
	return matrix


def mag(matrix):
	return np.sum(matrix)

def tot_energy(matrix):
	tot_e = 0
	for i in range(rows):
		tot_e += -1*(J*matrix[i]*(matrix[i-1])+h*matrix[i])#each pair of sites should be counted only once
	return tot_e



start_time = time.time()

J=1.
rows=32
h=0
T_c=0.65
steps=2**17
averageing_steps = 2**9

T_array = np.random.normal(T_c, 0.5, 200)
T_array = T_array[(T_array>0.2) & (T_array<1)]
num_temps = len(T_array)

energy = np.zeros(num_temps)
magnetization = np.zeros(num_temps)
specheat = np.zeros(num_temps)
magsuscep = np.zeros(num_temps)



initial_matrix = np.random.choice([-1,1],size=rows)


for i in range(num_temps):
	neg_beta = -1./T_array[i]
	E = np.zeros(averageing_steps)
	M = np.zeros(averageing_steps)
	f_matrix = metrop(initial_matrix, steps, neg_beta)
	
	for j in range(averageing_steps):
		f_matrix = metrop(f_matrix, rows*rows, neg_beta)
		E[j] = tot_energy(f_matrix)
		M[j] = mag(f_matrix)
	energy[i] = np.sum(E)
	magnetization[i] = abs(np.sum(M))
	specheat[i] = np.sum(E*E) - np.sum(E)*np.sum(E)/averageing_steps
	magsuscep[i] = np.sum(M*M) - np.sum(M)*np.sum(M)/averageing_steps

c = rows*averageing_steps

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
