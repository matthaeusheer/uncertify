import numpy as np
import math

def castInt(n):
    return np.int(np.round(n))

def floorInt(n):
    return np.int(np.floor(n))

def nCm(n,m):
    if m > n:
        return 0.0
    elif m == n:
        return 1.0
    elif m == 0:
        return 1.0
        
    RN = sum(np.log(range(1,n+1)))
    RM = sum(np.log(range(1,m+1)))
    RNM = sum(np.log(range(1,n-m+1)))
    return castInt(np.exp(RN - RM - RNM))


def binomial_value(n, N, p):
    return nCm(N,n)*np.power(p, n)*np.power(1-p, N-n)

def nCm_ratio(n1,m1,n2,m2):
    
    if m1 > n1:
        return 0.0
    elif m2 > n2:
        return 0.0
        
    RN1 = sum(np.log(range(1,n1+1)))
    RM1 = sum(np.log(range(1,m1+1)))
    RNM1 = sum(np.log(range(1,n1-m1+1)))
    RN2 = sum(np.log(range(1,n2+1)))
    RM2 = sum(np.log(range(1,m2+1)))
    RNM2 = sum(np.log(range(1,n2-m2+1)))
    
    return np.exp(RN1 - RM1 - RNM1 - RN2 + RM2 + RNM2)

def nCm2_ratio(n1,m1,n2,m2,n3,m3):
    if m1 > n1:
        return 0.0
    elif m2 > n2:
        return 0.0
    elif m3 > n3:
        return 0.0
    RN1 = sum(np.log(range(1,n1+1)))
    RM1 = sum(np.log(range(1,m1+1)))
    RNM1 = sum(np.log(range(1,n1-m1+1)))
    RN2 = sum(np.log(range(1,n2+1)))
    RM2 = sum(np.log(range(1,m2+1)))
    RNM2 = sum(np.log(range(1,n2-m2+1)))
    RN3 = sum(np.log(range(1,n3+1)))
    RM3 = sum(np.log(range(1,m3+1)))
    RNM3 = sum(np.log(range(1,n3-m3+1)))
        
    return np.exp(RN1 - RM1 - RNM1 + RN2 - RM2 - RNM2 - RN3 + RM3 + RNM3)