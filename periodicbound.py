#!/usr/bin/env python

"""
isingmat.py was taken and periodic boundary conditions were added
"""

import matplotlib.pyplot as plt
import numpy as np
import random

J=1.
columns=100
rows=100
k_b=1.
T=1.
h=0.


iterations=500

isingmat = np.zeros((rows,columns))

for i in range(rows):
    isingmat[i]=np.random.choice([1,-1],columns)
print isingmat



deltaE = np.zeros((rows,columns))

for k in range(iterations):
    for i in range(rows):
        for j in range(columns):
            deltaE[i][j]=-2*J*isingmat[i][j]*(isingmat[i-1][j]+isingmat[(i+1) % (rows)][j]+isingmat[i][j-1]+isingmat[i][(j+1) % (columns)])-2*h*isingmat[i][j]

    p_flip = np.exp((-deltaE)/(T))

    for i in range(rows):
        for j in range(columns):
            if p_flip[i][j]>random.random():
                isingmat[i][j]*=-1

print p_flip
print isingmat





plt.imshow(isingmat,cmap='Greys', interpolation = 'none')

plt.show()
