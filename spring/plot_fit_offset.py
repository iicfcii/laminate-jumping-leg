import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
import data
import process

ts = [32.5]
ls = [25,37.5,50,62.5,75]
ws = [10,20,30]

L,W,T = np.meshgrid(ls,ws,ts)
K = []
for l,w,t in zip(L.flatten(),W.flatten(),T.flatten()):
    k,b,data = process.k(t,w,l,samples=['1'],has_gap=False)
    K.append(k)
K = np.array(K).reshape(L.shape)

# Model
# k = E*w*t^3/(L+b)/16
# 1/k = 16*(l+b)/E*w*t^3
def model(l,w,t):
    l_wt3 = (l.flatten()*1e-3)/(w.flatten()*1e-3)/(t.flatten()*2.54e-5)**3
    wt3_inv = 1/(w.flatten()*1e-3)/(t.flatten()*2.54e-5)**3
    return l_wt3,wt3_inv

l_wt3,wt3_inv = model(L,W,T)
A = np.concatenate((
    l_wt3.reshape(-1,1),
    wt3_inv.reshape(-1,1)),
    axis=1
)
B = 1/K.flatten()
coeff,r,rank,s = np.linalg.lstsq(A,B,rcond=None)
E = 16/coeff[0]
b = coeff[1]/coeff[0]
print('E {:.2f} b {:.2f}'.format(E/1e9,b))

for z in range(len(ts)):
    plt.figure()
    for y in range(len(ws)):
        ks_inv = 1/K[y,:,z].flatten()

        l_wt3,wt3_inv = model(L[y,:,z],W[y,:,z],T[y,:,z])
        kps_inv = 16*l_wt3/E + 16*b*wt3_inv/E

        c = 'C{:d}'.format(y)
        plt.plot(L[y,:,z],ks_inv,'o',color=c,label='w={:.1f}mm'.format(ws[y]))
        plt.plot(L[y,:,z],kps_inv,'-',color=c)
    plt.xlabel('l [mm]')
    plt.ylabel('1/k [Nm/rad]')
    plt.legend()

for z in range(len(ts)):
    plt.figure()
    for x in range(len(ls)):
        ks = K[:,x,z].flatten()
        # kps = model(L[:,x,z],W[:,x,z],T[:,x,z],coeff)

        l_wt3,wt3_inv = model(L[:,x,z],W[:,x,z],T[:,x,z])
        kps = 1/(16*l_wt3/E + 16*b*wt3_inv/E)

        c = 'C{:d}'.format(x)
        plt.plot(W[:,x,z],ks,'o',color=c,label='l={:.1f}mm'.format(ls[x]))
        plt.plot(W[:,x,z],kps,'-',color=c)
    plt.xlabel('w [mm]')
    plt.ylabel('k [Nm/rad]')
    plt.legend()
plt.show()
