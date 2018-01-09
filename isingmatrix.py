#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import random

J=1
columns=100
rows=100
k_b=1
T=1

iterations=50


isingmat = np.zeros((rows,columns))

for i in range(rows):
    isingmat[i]=np.random.choice([1,-1],columns)
print isingmat


deltaE = np.zeros((rows-2,columns-2))


for i in range(iterations):
    for i in range(rows-2):
        for j in range(columns-2):
            deltaE[i][j]=2*J*isingmat[i+1][j+1]*(isingmat[i][j+1]+isingmat[i+2][j+1]+isingmat[i+1][j]+isingmat[i+1][j+2])

    p_flip = np.exp((-deltaE)/(k_b*T))

    for i in range(rows-2):
        for j in range(columns-2):
            if p_flip[i][j]>random.random():
                isingmat[i+1][j+1]*=-1

plt.imshow(isingmat, cmap='Greys')
plt.show()
	
