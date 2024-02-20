# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 08:31:26 2021

@author: bachynsk
"""

import numpy as np
import matplotlib.pyplot as plt

# generate a wind file for step wind. The file format is: 

#File head: (The file head may contain an arbitrary number of comment lines starting with:‘)
#Line 1 : Number of samples
#Line 2 : Time step
#Line 3 : Number of columns (=2)
#Time series:
#Line 4 : velocity direction
#Line 5-N : velocity direction


# The obtained file can be used with the "fluctuating two-component" wind
# option in SIMA. 

# inputs: 
fname = 'step_wind_long.asc' # output file
WS = np.array([4,6,8,10,11.4,12,14,16,18,20,22,24])
dt = 0.1
tWS = 800 # duration for each wind speed
tstart = 600 # extra time for first wind speed

# Compute the length of the file
nWS = len(WS) 
tTot = tWS*nWS + tstart

# generate time and wind vectors
t = np.arange(0,tTot,dt)
wvel = np.ones([len(t),1]) 
nstart = sum(t<tstart) 
wvel[0:nstart] = WS[0]*wvel[0:nstart]; 
ind1 = nstart 
npWS = int(tWS/dt) 

for ii in range(0,len(WS)):
    wvel[ind1:ind1+npWS] = WS[ii] 
    ind1 = ind1 + npWS 

# Plot the time series to check
plt.figure()
plt.plot(t,wvel,'k')
plt.xlabel('Time, s')
plt.ylabel('V_h_u_b, m/s')
plt.ylim(0,25)

# Write the time series to file
fid = open(fname,'w')
fid.write('%d\n' % len(t));
fid.write('%f\n' % dt); 
fid.write('%d\n' % 2); 
for ii in range(0,len(wvel)):
    fid.write('%f %f  \n' % (wvel[ii],0));
fid.close()
