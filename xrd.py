"""
Created on Tue Jan 31 12:34:00 2015
Author: Jerome C. Taguba
Description: XRD simulation
Updates:
1. Output file changed to .txt
2. Limit of x-axis changed to (0,85) due to sudden increase of intensity at theta=90 degrees
3. Normalized output intensity
4. Added automatic plot saving
5. Lattice parameters and basis imported from a csv file
Ready for version 2.0 (beta)

Bugs:
1. (hkl) with the same angle were not accounted
"""

import numpy as np
import math as m
import matplotlib.pyplot as plt
import time

#start timer
start_time = time.clock()

#lattice parameters and basis atoms
database = np.genfromtxt('Database/Ti3O_P312.csv', delimiter=',', dtype= float)
a = database[0,5]
b = database[0,6]
c = database[0,7]
alpha = database[0,8]
beta = database[0,9]
gamma = database[0,10]
basis = database

#xray wavelength
wavelength = 1.54056 #for Cu K alpha 

#maximum hkl
max_hkl = 15

#atomic form factor database
asf = np.genfromtxt('Database/asf.csv', delimiter=',', dtype= float)

#conversion
alpha = m.radians(alpha)
beta = m.radians(beta)
gamma = m.radians(gamma)


#functions
def V(a, b, c, alpha, beta, gamma):
    first = (a**2)*(b**2)*(c**2)
    second = (m.cos(alpha))**2 + (m.cos(beta))**2 + (m.cos(gamma))**2
    third = 2*m.cos(alpha)*m.cos(beta)*m.cos(gamma)
    return m.sqrt(first*(1 - second + third))

def F(alpha, beta, gamma):
    return m.cos(alpha)*m.cos(beta) - m.cos(gamma)

def d_hkl(h, k, l, a, b, c, alpha, beta, gamma):
    first = (h**2)*(b**2)*(c**2)*(m.sin(alpha))**2 + (k**2)*(a**2)*(c**2)*(m.sin(beta))**2 + (l**2)*(a**2)*(b**2)*(m.sin(gamma))**2
    second = 2*h*k*a*b*(c**2)*(F(alpha, beta, gamma)) + 2*k*l*(a**2)*b*c*(F(beta, gamma, alpha)) + 2*l*h*a*(b**2)*c*(F(gamma, alpha, beta))
    g = (1.0/V(a, b, c, alpha, beta, gamma))*m.sqrt(first + second)
    return 1.0/abs(g)
    
def structure_factor(d, asf, element):
    s = 1.0/(2.0*d)
    first = asf[element,1]*m.exp(-asf[element,2]*(s**2))
    second = asf[element,3]*m.exp(-asf[element,4]*(s**2))
    third = asf[element,5]*m.exp(-asf[element,6]*(s**2))
    fourth = asf[element,7]*m.exp(-asf[element,8]*(s**2))
    return first + second + third + fourth + asf[element,9]

def intensity(h, k, l, basis, d, asf, theta):
    cos_sum = 0
    sin_sum = 0
    for atoms in basis:
        P = (1 + (m.cos(2*theta))**2)/(((m.sin(theta))**2)*(m.cos(theta)))
        S = structure_factor(d, asf, atoms[0])
        correction = P*S
        cos_sum = cos_sum + correction*m.cos(2*m.pi*(h*atoms[1] + k*atoms[2] + l*atoms[3]))
        sin_sum = sin_sum + correction*m.sin(2*m.pi*(h*atoms[1] + k*atoms[2] + l*atoms[3]))
    intensity = cos_sum**2 + sin_sum**2
    return intensity



#counts the proper array size
print("calculating data size...")
counter = 0
for h in range(-max_hkl, max_hkl + 1):
    for k in range(-max_hkl, max_hkl + 1):
        for l in range(-max_hkl, max_hkl + 1):
            counter += 1


#main program: calculation of intensity
print("generating hkl and 2theta...")
data = np.zeros([counter,11])
index1 = 0
for h in range(-max_hkl, max_hkl + 1):
    for k in range(-max_hkl, max_hkl + 1):
        for l in range(-max_hkl, max_hkl + 1):
            if (abs(h) + abs(k) + abs(l)) != 0:
                d = d_hkl(h, k, l, a, b, c, alpha, beta, gamma)
                sin_theta = 0.5*wavelength*(1.0/d)
                if abs(sin_theta) < 0.70:
                    theta = m.asin(sin_theta)
                    two_theta = m.degrees(2*theta)
                    sum_hkl1 = abs(h) + abs(k) + abs(l)
                    sum_hkl5 = abs(h)**5 +abs(k)**5 + abs(l)**5                   
                    I = intensity(h, k, l, basis, d, asf, theta)                 
                    if I > 10**(-7):
                        data[index1,0] = abs(h)
                        data[index1,1] = abs(k)
                        data[index1,2] = abs(l)
                        data[index1,3] = round(d,8)#rounded to minimum error
                        data[index1,4] = sum_hkl1 #not yet useful
                        data[index1,5] = sum_hkl5 #not yet useful
                        data[index1,6] = round(sin_theta,8) #rounded to minimum error
                        data[index1,7] = round(two_theta,8) #in degrees, rounded to minimum error
                        data[index1,8] = I
                        index1 += 1

#removes zero rows
print("removing zero rows...")
data = data[~(data[:,4] == 0)]

#multiplicity
print("counting multiplicities...")
for index2 in range(len(data[:,0])):
    multiplicity = 0
    for index3 in range(len(data[:,0])):
        if (data[index2,3] == data[index3,3] and data[index2,4] == data[index3,4] and data[index2,5] == data[index3,5]):
            multiplicity += 1
    data[index2,9] = multiplicity
    data[index2,10] = multiplicity * data[index2,8]

#unique values
print("looking for unique values...")
unique_data = np.zeros([len(data[:,0]),11])
index4 = 0
for index5 in range(len(data[:,0])):
    if data[index5,3] not in unique_data[:,3]:
        for index6 in range(len(data[0,:])):
            unique_data[index4, index6] = data[index5,index6]
        index4 += 1
data = unique_data
data = data[~(data[:,4] == 0)] 


#lorentian plotting
print("generating lorentian plot...")
deviation = 0.005
size = 85.0/deviation + 1
xydata = np.zeros([size,2])
index = 0
for x_axis in np.linspace(0,85,int(size)):
    xydata[index,0] = x_axis
    index += 1
for index_data in range(len(data[:,0])):
    for index_xyplot in range(len(xydata[:,0])):
        D = 10
        w =(0.9*wavelength)/(D*m.cos(m.radians(xydata[index_xyplot,0])))
        denominator = 1 + 4*((xydata[index_xyplot, 0] - data[index_data,7])/w)**2
        xydata[index_xyplot,1] = xydata[index_xyplot,1] + (float(data[index_data, 10])/denominator)
xydata[:,1] = 100*xydata[:,1]/max(xydata[:,1])
plt.plot(xydata[:,0], xydata[:,1])
plt.xlim(0,85)
plt.ylim(0,105)
plt.savefig('plot.png', dpi = 200, transparent = True )


#file saving
print("saving...")
file = open("data_peaks.txt", "w")
for rows in data:
    for elements in rows:
        file.write(str(elements) + "\t")
    file.write("\n")
file.close()

file = open("data_plot.txt", "w")
for rows in xydata:
    for elements in rows:
        file.write(str(elements) + "\t")
    file.write("\n")
file.close()

print("simulation done!")
print("run time: " + str(time.clock() - start_time))