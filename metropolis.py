#!/usr/bin/env python

"""
Here periodicbound.py was taken and the original algorithm was replaced with the metropolis algorithm
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import time

start_time = time.time()

J=1.
columns=50
rows=50
T=1.
h=0

iterations=2**16

isingmat = np.random.choice([-1,1],size=(rows,columns))



k=0

deltaE=0
i=0
j=0
spin_change=0
equilib = []
neg_beta = -1./T



while k < iterations:
    i = np.random.randint(0,rows)
    j = np.random.randint(0,columns)
	
    deltaE=2*J*isingmat[i][j]*(isingmat[i-1][j]+isingmat[(i+1) % (rows)][j]+isingmat[i][j-1]+isingmat[i][(j+1) % (columns)])+2*h*isingmat[i][j]

    if deltaE<=0:
        isingmat[i][j] *= -1
        spin_change += 1
    elif random.random() < np.exp(deltaE*neg_beta):
        isingmat[i][j] *= -1
        spin_change += 1
    equilib.append(spin_change)
    k+=1


if iterations>=10000:
    equilib = equilib[::iterations/10000]



print isingmat

plt.imshow(isingmat,cmap='Greys')

plt.figure()


plt.plot(equilib)#this graph illustrates how long it took for the system to reach equilibrium
print "%s seconds" % (time.time() - start_time)
print np.mean(isingmat)
plt.show()
