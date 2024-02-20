# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 15:49:20 2021

@author: bachynsk
"""

import numpy as np
import matplotlib.pyplot as plt


def turningpoints(lst):
    dx = np.diff(lst)
    tps_dx = (dx[1:] * dx[:-1] < 0)
    tps = np.array([1], dtype=bool)
    tps1 = np.append(tps,tps_dx)
    tp = np.append(tps1,tps)
    return tp

def PQanalysisFun(t,x,plotflag): 
    # PQ analysis
    # Inputs: time series t and motion history x. Note that x should have zero mean! 
    # Outputs: coefficients P and Q, raw data xbar and dx, and indices tp for 
    # all turning points (includes negatives!)

    if plotflag==1:
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(t,x)
        plt.xlabel('t')
        plt.ylabel('x')
    
    tp = turningpoints(x) # find turning point indices
    ttp1 = t[tp]
    xtp1 = x[tp]
    
    # select only positive turning points
    ttp = ttp1[xtp1>0]
    xtp = xtp1[xtp1>0]
    
    n = len(xtp)
    
    if n<3:
        print('Error from PQanalysis: Not enough turning points identified')
        return 0
    
    if plotflag==1:
        plt.plot(ttp,xtp,'k.')
        
    # find the mean and differences
    xbar = 0.5*( xtp[1:n-1]+xtp[0:n-2])
    dx = ( xtp[0:n-2]-xtp[1:n-1])/xbar
    
    # linear fit 
    pcoeffs = np.polynomial.polynomial.polyfit(xbar,dx,1)
    P = pcoeffs[0] # P coefficient
    Q = pcoeffs[1] # Q coefficient
    
    if plotflag ==1:
        plt.subplot(1,2,2)
        plt.plot(xbar,dx,'b.')
        plt.plot(xbar,np.polynomial.polynomial.polyval(xbar,pcoeffs))
        plt.ylim(0,0.3)
    
    return P,Q,xbar,dx,tp
    