#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from matplotlib import animation




start_time = time.time()

J=1.
N=20
N2=2*N
T=1.
h=0
neg_beta = -1./T

iterations=2**4

matrix = np.random.choice([-1,1],size=(N,N))

fig = plt.figure(figsize=(20,20))

k=0

deltaE=0
i=0
j=0
spin_change=0
equilib = []
img = []



patches = []

for i in range(N2):
    for j in range(N):
        patches.append(plt.Circle((j,i), radius=.3, lw=2, ec="black", facecolor=None))
        
def init():
    #Draw background
    im = plt.imshow(np.zeros((N2,N)),interpolation="none", hold=True)
    axes = plt.gca()
    axes.autoscale(False)

    #Add patches to axes
    for p in patches:
        axes.add_patch(p)
    return [im]
        
def colour_matrix(isingmat):
	return (isingmat + 1)/2
	
	
def animate(n):
	i=0
	j=0
	for i in range(N2):
		for j in range(N):
			if (i+j)%2:
				patches[i*N + j].set_visible(False)
			else:
				patches[i*N + j].set_facecolor(str(colours[n][i/2][j]))
				patches[i*N + j].set_visible(True)
	return patches


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
   	if k%1 == 0:
		img.append(matrix)


#equilib = equilib[::iterations/10000]

colours = []

for i in range(len(img)):
	colours.append(colour_matrix(img[i]))



anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(img), blit=True)

anim.save('test.mp4', writer="ffmpeg", fps=2)



print matrix

#plt.plot(equilib)



print "%s seconds" % (time.time() - start_time)
plt.show()