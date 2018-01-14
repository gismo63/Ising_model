#!/usr/bin/env python
"""
This program uses FuncAnimation to animate the state of the lattice at equally
spaced intervals throughout its progression
"""
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
initial = isingmat


k=0

deltaE=0
i=0
j=0
spin_change=0#keeps track of the amount of spin changes
equilib = []
frames = 1000#number of frames in the animation
img = np.ndarray((frames,rows,columns), dtype = 'float')#here an array had to be used because appending to a list was causing bugs


count = 0
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
        img[count] = isingmat
        count+=1

if iterations>=10000:
    equilib = equilib[::iterations/10000]#since there are so many spin changes especially for larger latices only some of the values in thelist are chosen so as not to get a memory error


def init():
    return [plt.imshow(initial,cmap='Greys',interpolation='none')]

def animate(n):
    im = plt.imshow(img[n],cmap='Greys',interpolation='none')
    return [im]

fig=plt.figure()

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(img), interval = 1, blit=True)#creates an animation by running a function a number of times

#plt.imshow(,cmap='Greys', interpolation = 'none')


print img

plt.plot(equilib)



print "%s seconds" % (time.time() - start_time)
plt.show()

