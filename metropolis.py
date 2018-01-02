#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import random
import time

start_time = time.time()

J=1.
columns=30
rows=30
T=2.2
h=0

iterations=1000000

isingmat = np.random.choice([-1,1],size=(rows,columns))



k=0

deltaE=0
i=0
j=0
spin_change=0
equilib = []


while k < iterations:
	
	i = random.choice(np.arange(rows))
	j = random.choice(np.arange(columns))
	
	deltaE=2*J*isingmat[i][j]*(isingmat[i-1][j]+isingmat[(i+1) % (rows)][j]+isingmat[i][j-1]+isingmat[i][(j+1) % (columns)])+2*h*isingmat[i][j]

	if deltaE<=0:
		isingmat[i][j] *= -1
		spin_change += 1
	elif random.random() < np.exp(-deltaE/T):
		isingmat[i][j] *= -1
		spin_change += 1
	equilib.append(spin_change)
	k+=1

	if k%10000 == 0:
		plt.imshow(isingmat,cmap='Greys')
		plt.draw()
		plt.pause(0.0001)

equilib = equilib[::iterations/10000]




#plt.plot(np.exp(np.array(equilib)*1e-5))


print isingmat

plt.imshow(isingmat,cmap='Greys')

plt.figure()


plt.plot(equilib)
print "%s seconds" % (time.time() - start_time)
plt.show()
