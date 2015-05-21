# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:17:40 2015

@author: xzhao
"""

#import math
import numpy as np
import matplotlib.pyplot as plt

#tmpA = np.loadtxt("u_profile_0.000000_500.txt")
#x = tmpA[:,0]
#y = tmpA[:,1]
#
## plot the data
#plt.figure(figsize=(10,5), dpi=200)
#plt.plot(x,y,label="$u$", color="red", linewidth=2)
#
#plt.xlabel("y")
#plt.ylabel("velocity u")
##plt.xlim(0.0, 10)
##plt.ylim(0.0, 10)
#plt.title("velocity profile at the cross section")
#plt.legend()
#plt.show()


# -------------------------------------------------------------------
# read and plot the history data
tmpA = np.loadtxt("u_profile_history_-50.000000.txt")
mA = tmpA.shape[0]  # mA = 100 here
nA = tmpA.shape[1]
#mA = 50

# y coordinates of cross section
y = tmpA[0,:]
ts = tmpA[:,0]

# time steps at which we would like to plot
tsplot = np.array([ 10, 50, 60, 100, 300 ])
colorplot = ["red", "blue", "green", "cyan", "black" ]
labelplot = [ "t = "+str( x ) for x in tsplot ]

plt.figure(figsize=(10,5), dpi=100)

for i in range(1,mA):
    for j in range(0,len(tsplot)):
        #print "i=",i,"j=",j, ", ts[t] = ", tsplot[j], ", t = ", ts[i]
        if  tsplot[j]==tmpA[i,0]:
            #print "ts[t] = ", tsplot[j], ", t = ", ts[i]
            vel_u = tmpA[i,:]
            plt.plot(y[2:nA],vel_u[2:nA],label=labelplot[j], color=colorplot[j], linewidth=2)

# figure setup
plt.xlabel("y")
plt.ylabel("velocity u")
plt.title("velocity profile at the cross section")
plt.legend()
plt.show()


# -------------------------------------------------------------------
#plot velocity history of a point w.r.t. time
y0 = 0.0
#mA = 50

time = tmpA[1:mA,1]
for i in range(2,nA):
    if tmpA[0,i]==y0:
        ut = tmpA[1:mA,i]
        break

plt.figure(figsize=(10,5), dpi=100)
plt.plot(time,ut,label="u (x = 0.0, y = 0.0)", color="black", linewidth=2)
plt.xlabel("time (s)")
plt.ylabel("velocity u")
plt.title("velocity history")
plt.legend()
plt.show()
