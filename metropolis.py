#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import random

J=1.
columns=200
rows=200
T=2.3
h=0

iterations=10000000

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

equilib = equilib[::100]

plt.plot(equilib)


#plt.plot(np.exp(np.array(equilib)*1e-5))

plt.figure()

print isingmat

plt.imshow(isingmat,cmap='Greys')
plt.show()
