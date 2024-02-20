# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:01:24 2022

@author: seragela
"""

import numpy as np
import rainflow as rfc

def FatigueDamage(stress_history,thk,thk_ref,k,K1,beta1,stress_lim=0,K2=0,beta2=0):
    cc = rfc.count_cycles(stress_history)
    sk = [c[0] for c in cc] # stress range
    n = [c[1] for c in cc] # cycle count
    
    Ns = np.zeros(len(sk)) #initialize damage
    
    for i,s in enumerate(sk):
        if s>stress_lim:
            beta = beta1; K = K1;
        else:
            beta = beta2; K = K2;
        
        Ns[i] = 1/K*(s*(thk/thk_ref)**0.2)**(-beta)
        
    FD = np.sum(n/Ns)
    
    return FD   

    
    
            

