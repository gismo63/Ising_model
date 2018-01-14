#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from matplotlib import animation




start_time = time.time()

J=1.
columns=20
rows=20
T=1.
h=0

iterations=1000000

isingmat = np.random.choice([-1,1],size=(rows,columns))


k=0

deltaE=0
i=0
j=0
spin_change=0#keeps track of the amount of spin changes
equilib = []
frames = 1000#number of frames in the animation
img = []#will be a list of matrix plots



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

    if k%(iterations/frames) == 0:#takes a snapshot of the state of the matrix at equally spaced intervals such that the final animation will have $frames frames
        img.append([plt.imshow(isingmat,cmap='Greys')])

if iterations>=10000:
    equilib = equilib[::iterations/10000]#since there are so many spin changes especially for larger latices only some of the values in thelist are chosen so as not to get a memory error




fig=plt.figure()

ani = animation.ArtistAnimation(fig, img, interval = 0, blit = True, repeat_delay = 1000)#creates an animation out of the img list




print isingmat

plt.plot(equilib)



print "%s seconds" % (time.time() - start_time)
plt.show()

