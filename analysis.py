#!/usr/bin/env python

"""
This program calculates the magnetization, energy, heat capacity and magnetic susceptibility
per site of the 2d square lattice at various temperatures around the critical temperature
and plots graphs of them
"""


import matplotlib.pyplot as plt
import numpy as np
import random
import time



def metrop(matrix, iterations, neg_beta):#metropolis algorithm
    k=0
    while k < iterations:#using a while loop becuase it's slightly more efficient
        #pick a random coordinate
        i = np.random.randint(0,rows)
        j = np.random.randint(0,columns)
        #calculate delta E
        deltaE=2*J*matrix[i][j]*(matrix[i-1][j]+matrix[(i+1) % (rows)][j]+matrix[i][j-1]+matrix[i][(j+1) % (columns)])+2*h*matrix[i][j]
        
        if deltaE<=0:#if the change in energy is less than 0 then the probability will be greater than 1 so can change immmediately, this is slightly more efficient than just having the second statement
            matrix[i][j] *= -1
        elif random.random() < np.exp(deltaE*neg_beta):#if the change in energy is greater than 0 then the probability of a flip will be between 0 and 1 so a random number between 0 and 1 is generated and if that number is less than the probability then the spin if flipped
            matrix[i][j] *= -1
        k+=1
    return matrix


def mag(matrix):
    return np.sum(matrix)#returns the total magnetization of the given lattice

def tot_energy(matrix):
    tot_e = 0
    for i in range(rows):
        for j in range(columns):
            #here only the interation with the spin above and the spin to the left are counted, this way no interaction is counted twice and this is a lot more efficient than counting all nearest neighbour interactions for each spin and dividing by 2
            tot_e += -1*(J*matrix[i][j]*(matrix[i-1][j]+matrix[i][j-1])+h*matrix[i][j])#each pair of sites should be counted only once
    return tot_e



start_time = time.time()#records the time when the program starts


#these variables can all be changed, program took me around 10 minutes to run with current config
J=1.#interaction strength
columns=16#columns in the matrix
rows=16#rows in the matrix
h=0#magnetic field strength
T_c=2.2692#approximate value for critical temp
steps=2**18#number of steps to attempt reach equilibrium
averageing_steps = 2**10#number of configs to average over
num_temps = 300#number of temperatures to analyse

T_array = np.random.normal(T_c, 0.5, num_temps)#since the most important temperatures to calculate are those around the critical temperature a normal distribution of the termperatures is used here
T_array = T_array[(T_array>1.5) & (T_array<3.5)]#any outliers are removed from the array
num_temps = len(T_array)

#better to define numpy arrays and than to append to a list
energy = np.zeros(num_temps)
magnetization = np.zeros(num_temps)
specheat = np.zeros(num_temps)
magsuscep = np.zeros(num_temps)



initial_matrix = np.random.choice([-1,1],size=(rows,columns))#creat a random matrix with values of -1 and 1


for i in range(num_temps):
    neg_beta = -1./T_array[i]#more efficient to calculate beta outside the loop and multiply in the loop ranther than divide in the loop
    E = np.zeros(averageing_steps)#values of energy and magnetism to average as they are quite random above the critical temperature
    M = np.zeros(averageing_steps)
    f_matrix = metrop(initial_matrix, steps, neg_beta)#attempt to reach equillibrium
	
    for j in range(averageing_steps):
        f_matrix = metrop(f_matrix, rows*columns, neg_beta)#perform a few more iterations in order to create different configurations to average over
        E[j] = tot_energy(f_matrix)
        M[j] = mag(f_matrix)
    energy[i] = np.sum(E)#averaging is done outside the loops for efficiency
    magnetization[i] = abs(np.sum(M))
    specheat[i] = np.sum(E*E) - np.sum(E)*np.sum(E)/averageing_steps
    magsuscep[i] = np.sum(M*M) - np.sum(M)*np.sum(M)/averageing_steps

c = rows*columns*averageing_steps

energy = energy / (c)
magnetization = magnetization / (c)
specheat = (specheat / (T_array**2)) / (c)
magsuscep = (magsuscep / (T_array)) / (c)


plt.plot(T_array, magnetization, 'o')

plt.figure()

plt.plot(T_array, energy, 'o')

plt.figure()

plt.plot(T_array, specheat, 'o')

plt.figure()

plt.plot(T_array, magsuscep, 'o')


print f_matrix

print "%s seconds" % (time.time() - start_time)#runtime for the program
plt.show()
