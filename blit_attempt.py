#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from matplotlib import animation




start_time = time.time()

J=1.
columns=200
rows=200
T=1.
h=0

iterations=1000000

isingmat = np.random.choice([-1,1],size=(rows,columns))

im = plt.imshow(isingmat,cmap='Greys', animated=True)

k=0

deltaE=0
i=0
j=0
spin_change=0
equilib = []
img = []



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
		img.append([plt.imshow(isingmat,cmap='Greys')])


equilib = equilib[::iterations/10000]




fig=plt.figure()

ani = animation.ArtistAnimation(fig, img, interval = 0, blit = True, repeat_delay = 1000)




print isingmat

plt.plot(equilib)



print "%s seconds" % (time.time() - start_time)
plt.show()