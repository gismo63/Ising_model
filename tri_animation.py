#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from matplotlib import animation




start_time = time.time()

J=1.
N=30
N2=2*N
T=1.
h=0
neg_beta = -1./T

frames = 2**8

iterations=2**16

matrix = np.random.choice([-1,1],size=(N,N))
i_mat = matrix

fig = plt.figure(figsize=(20,20))

k=0

deltaE=0
i=0
j=0
spin_change=0
equilib = []
img = np.ndarray((frames,N,N), dtype = 'float')



patches = []

for i in range(N):
    for j in range(N2):
        patches.append(plt.Circle((j,i), radius=.5, facecolor=None))
        
def init():
    #Draw background
    plt.imshow(np.zeros((N,N2)), animated=True)
    axes = plt.gca()
    axes.autoscale(False)

    #Add patches to axes
    for p in patches:
        axes.add_patch(p)
    return patches
        
def colour_matrix(isingmat):
	return (isingmat + 1)/2
	
	
def animate(n):
    i=0
    j=0
    for i in range(N):
        for j in range(N2):
            if (i+j)%2:
                patches[i*N2 + j].set_visible(False)
            else:
                patches[i*N2 + j].set_facecolor(str(colours[n][i][j/2]))
                patches[i*N2 + j].set_visible(True)
    return patches

count = 0
while k < iterations:
    i = np.random.randint(0,N)
    j = np.random.randint(0,N)
    if i%2:
        deltaE=2*(J*matrix[i][j]*(matrix[i][j-1]+matrix[i][(j+1)%N]+matrix[i-1][(j+1)%N]+matrix[i-1][j]+matrix[(i+1)%N][(j+1)%N]+matrix[(i+1)%N][j])+h*matrix[i][j])
    else:
        deltaE=2*(J*matrix[i][j]*(matrix[i][j-1]+matrix[i][(j+1)%N]+matrix[i-1][j-1]+matrix[i-1][j]+matrix[(i+1)%N][j-1]+matrix[(i+1)%N][j])+h*matrix[i][j])
    if deltaE<=0:
        matrix[i][j] *= -1
        spin_change += 1
    elif random.random() < np.exp(deltaE*neg_beta):
        matrix[i][j] *= -1
        spin_change += 1
    equilib.append(spin_change)
    k+=1
    if k%(iterations/frames) == 0:
        img[count] = matrix
        count +=1
        
print frames - count


#equilib = equilib[::iterations/10000]

colours = []

for i in range(len(img)):
	colours.append(colour_matrix(img[i]))



anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(img), interval = 1, blit=True)

#anim.save('test.mp4', writer="ffmpeg", fps=2)



#print img

#plt.plot(equilib)



print "%s seconds" % (time.time() - start_time)
plt.show()
