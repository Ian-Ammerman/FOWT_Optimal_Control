# -*- coding: utf-8 -*-
"""
ReadTimeDomainResults
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

#============================================================================#
# Inputs
SIMAfol = "C:\\simaWS\\IDAWT_v44\\SPAR2\\testturb\\"
#Total simulation time and storage time step
NtRiflexForce = 40000 # number of stored time steps on the force file (=simulationTime/dt)
hullBody = 1 # this is the number of the hull body (order in the SIMA inputs)

#============================================================================#
#============================================================================#

def unpackBinaryResults(fileName,Nt,Nchan):
    #Read .bin file
    # forfilebin is the name of the binary file
    # Nt is the number of time steps (if 0, then Nchan is needed)
    # Nchan is an optional argument - unused if Nt>0

    with open(fileName, mode='rb') as file:
        fileContent = file.read()
        
    #Reshape results in file to a 2D array 'A', with one result component per row and one time step per column 
    numRecords = int((len(fileContent) / 4))
    unpacked = struct.unpack("f" * numRecords ,fileContent)
    if Nt>1:
        Nchan = len(np.asarray(unpacked))/Nt
    else:
        Nt = len(np.asarray(unpacked))/Nchan
    # print(Nchan,Nt,numRecords)
    A = np.reshape(np.asarray(unpacked), (int(Nchan), int(Nt)), order='F')       
    
    
    return A 


def readSIMO_resultstext(filename):
    # read the SIMO text 

    chanNames = [];
    nchan = 0;
    nts = 0;
    dt = 0;
    with open(filename,'r') as f: 
    
        # read the header
    
        for ii in range(0,6): 
            tline = f.readline()
        #number of samples
        tline = f.readline()
        regexp = r"(\d+)"
        d = np.fromregex(StringIO(tline), regexp,[('num',np.int64)])
        nts = int(d['num']) # number of time steps
        tline2 = f.readline()
        tsplit = tline2.split(' ')
        dt = float(tsplit[8]) # time step
        tline = (f.readline()).split(' ')
        tstart = float(tline[7])
        tline = (f.readline()).split(' ')
        tend = tline[9]
    
        for tline in f.readlines():
            if tline[0] != '*':
                nchan = nchan+1
                chanNames.append(tline.split())


    return nchan, nts, dt, chanNames

def getchannelNumbers(chanNames,B1): 
    chanMotions = 0
    ind1 = 0
    chanWave = 0
    chanAcc = 0
    nameMotion = 'B%dr29c1'% B1
    
    for ii in range(0,len(chanNames)): 
        xstr = chanNames[ii][0]
        x = xstr.find(nameMotion)
        if x>-1:  
            ind1 = ii
            chanMotions = ind1 + np.arange(0,6)


    
    ind1 = 0
    nameWave = 'Totalwaveelevation';
    for ii in range(0,len(chanNames)): 
        xstr = chanNames[ii][0]
        x = xstr.find(nameWave)
        if x>-1:  
            ind1 = ii
            chanWave=ind1 
    
    
    ind1 = 0
    nameAcc = 'B%dr30c1'%B1
    for ii in range(0,len(chanNames)): 
        xstr = chanNames[ii][0]
        x = xstr.find(nameAcc)
        if x>-1:  
            ind1 = ii
            chanAcc=ind1 

    return chanMotions, chanWave, chanAcc

def unpackSIMOresults(fileName,nts):
    with open(fileName, mode='rb') as file:
        fileContent = file.read()
    numRecords = int((len(fileContent) / 4))
    cols = len(np.asarray(struct.unpack("f" * numRecords, fileContent))) / nts
    A = np.transpose(np.reshape(np.asarray(struct.unpack("f" * numRecords, fileContent)), (nts, int(cols)), order='F'))

    return A

#---------------------------------------------RIFLEX results---------------------------------------------#
# Forces 
#Path + file name
ForcefileName = SIMAfol + 'sima_elmfor.bin'
A = unpackBinaryResults(ForcefileName,NtRiflexForce,0)


#Time vector
time_RIFLEX = A[1]

#NB! Forces from RIFLEX are in kN!

# Extract forces from tower and blade roots, assuming that force storage is 
# specified with tower base, then tower top, then mooring lines  

TowerBaseAxial   = A[2]  #Axial force
TowerBaseTors    = A[3]  #Torsional moment
TowerBaseBMY     = A[4]  #bending moment in local y
TowerBaseBMZ     = A[6]  #bending moment in local z
TowerBaseShearY  = A[8]  #shear force in local y
TowerBaseShearZ  = A[10] #shear force in local z
    
TowerTopAxial    = A[12]
TowerTopTors     = A[13]
TowerTopBMY      = A[15] # end 2
TowerTopBMZ      = A[17]
TowerTopShearY   = A[19]
TowerTopShearZ   = A[21]    
    
bl1Axial         = A[22] 
bl1Tors          = A[23]
bl1BMY           = A[24]
bl1BMZ           = A[26]
bl1ShearY        = A[28]
bl1ShearZ        = A[30]
    
bl2Axial         = A[32]
bl2Tors          = A[33]
bl2BMY           = A[34]
bl2BMZ           = A[36]
bl2ShearY        = A[38]
bl2ShearZ        = A[40]    


# wind turbine results
WTfileName = SIMAfol + 'sima_witurb.bin'

A = unpackBinaryResults(WTfileName,0,26)

time_WT    = A[1] # it is possible that this time vector differs from t
omega  = A[2]*np.pi/180; # convert from deg/s to rad/s
genTq  = A[4] 
genPwr = A[5] 
azimuth = A[6] 
HubWindX = A[7]
HubWindY = A[8]
HubWindZ = A[9]
AeroForceX = A[10]
AeroMomX = A[13]
Bl1Pitch = A[16]
Bl2Pitch = A[17]
Bl3Pitch = A[18] 

#---------------------------------------------SIMO results---------------------------------------------#
# #Path + file name
fileName = SIMAfol + 'results.tda'
fileNametxt = SIMAfol + 'results.txt'
nchan, nts, dt, chanNames = readSIMO_resultstext(fileNametxt)

AA = unpackSIMOresults(fileName,nts) 
# % Determine which channels to read for the platform motions, wave elevation
chanMotions, chanWave, chanAcc = getchannelNumbers(chanNames,hullBody)


time_SIMO = AA[1,:]
# summarize data in matrix
PlatMotions = AA[chanMotions,:]
wave = AA[chanWave,:]

#--------------------------------------------end read section------------------------------------------#
#------------------------------------------------------------------------------------------------------#


