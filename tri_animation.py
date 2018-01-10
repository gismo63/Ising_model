#!/usr/bin/env python

"""
This program animates the triangular lattice by using circular patches on a graph
Since the points in a triangular lattice are not directly above each other like in
the square lattice, this means that a simple imshow is not sufficient in this case
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
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
img = np.ndarray((frames,N,N), dtype = 'float')#here an array had to be used because appending to a list was causing bugs



patches = []#the patches must be stored in a list as this is what the FuncAnimation module takes in

for i in range(N):
    for j in range(N2):
        patches.append(plt.Circle((j,math.sqrt(3)*i), radius=1, facecolor=None))#Generate the coordinates for the circles on the graph, the sqrt3 is so that all nearest neighbours are the same distance, radius 1 ensures that the circles are touching
        
def init():#function that creates the initial state of the graph
    #Create a blue background so that the white and black circles can be clearly seen
    plt.imshow(np.zeros((round(math.sqrt(3)*N),N2)))
    axes = plt.gca()#create axis to add the patches to
    axes.autoscale(False)

    #Add patches to axes
    for p in patches:
        axes.add_patch(p)
    return patches
        
def colour_matrix(isingmat):# takes an ising matrix and returns a matrix with all -1s changed to 0s, This is becuase color='0' corresponds to white and color = '1' corresponds to black
	return (isingmat + 1)/2
	
	
def animate(n):
    i=0
    j=0
    for i in range(N):
        for j in range(N2):
            if (i+j)%2:
                patches[i*N2 + j].set_visible(False)#half of the points on the graph aren't needed so they are set invisible
            else:
                patches[i*N2 + j].set_facecolor(str(colours[n][i][j/2]))#set the facecolour to white or black
    return patches

count = 0
while k < iterations:
    i = np.random.randint(0,N)
    j = np.random.randint(0,N)
    if i%2:#Since the rowsn in a triangular lattice are not alligned the nearest neighbours differ based on whether it is an odd or even row
        deltaE=2*(J*matrix[i][j]*(matrix[i][j-1]+matrix[i][(j+1)%N]+matrix[i-1][(j+1)%N]+matrix[i-1][j]+matrix[(i+1)%N][(j+1)%N]+matrix[(i+1)%N][j])+h*matrix[i][j])
    else:
        deltaE=2*(J*matrix[i][j]*(matrix[i][j-1]+matrix[i][(j+1)%N]+matrix[i-1][j-1]+matrix[i-1][j]+matrix[(i+1)%N][j-1]+matrix[(i+1)%N][j])+h*matrix[i][j])
    if deltaE<=0:
        matrix[i][j] *= -1
    elif random.random() < np.exp(deltaE*neg_beta):
        matrix[i][j] *= -1
    k+=1
    if k%(iterations/frames) == 0:
        img[count] = matrix
        count +=1
        



colours = []

for i in range(len(img)):
	colours.append(colour_matrix(img[i]))
 



anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(img), interval = 1, blit=True)#creates an animation by running a function a number of times






print "%s seconds" % (time.time() - start_time)
plt.show()
