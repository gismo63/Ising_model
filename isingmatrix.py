#!/usr/bin/env python

import numpy as np

J=0.1
columns=4
rows=4

isingmat = np.zeros((rows,columns))

for i in range(rows):
	isingmat[i]=np.random.choice([1,-1],columns)
print isingmat


deltaE = np.zeros((rows-2,columns-2))
for i in range(rows-2):
	for j in range(columns-2):
		deltaE[i][j]=-2*J*isingmat[i+1][j+1]*(isingmat[i][j+1]+isingmat[i+2][j+1]+isingmat[i+1][j]+isingmat[i+1][j+2])

print deltaE
		
