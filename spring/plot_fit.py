import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
import data
import process

ts = [15,30]
ls = [25,50,75]
ws = [10,20,30]

# Simple beam model
# k = E*w*t^3/L/16
def model(l,w,t,coeff):
    return coeff*((w.flatten()/1000)*(t.flatten()*2.54e-5)**3/(l.flatten()/1000))

L,W,T = np.meshgrid(ls,ws,ts)

K = []
for l,w,t in zip(L.flatten(),W.flatten(),T.flatten()):
    k,b,data = process.k(t,w,l)
    K.append(k)
K = np.array(K).reshape(L.shape)

r = np.arange(0,1) # Use 15mill data
a = model(L[:,:,r],W[:,:,r],T[:,:,r],1).reshape(-1,1)
b = K[:,:,r].flatten()
coeff,r,rank,s = np.linalg.lstsq(a,b,rcond=None)
print('Fit Youngs modulus [GPa] {:.2f}'.format(coeff[0]*16/1e9))

for z in range(len(ts)):
    plt.figure()
    for y in range(len(ws)):
        ks = K[y,:,z].flatten()
        kps = model(L[y,:,z],W[y,:,z],T[y,:,z],coeff)

        c = 'C{:d}'.format(y)
        plt.plot(L[y,:,z],ks,color=c,label='w={:d}mm'.format(ws[y]))
        plt.plot(L[y,:,z],kps,'--',color=c)
    plt.xlabel('l [mm]')
    plt.ylabel('k [Nm/rad]')
    plt.legend()
plt.show()
