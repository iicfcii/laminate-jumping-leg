import sys
sys.path.append('../utils')

import multiprocessing
import matplotlib.pyplot as plt
import jump
import numpy as np
import data

cs = jump.cs

def sim(params):
    cs['k'] = params[0]
    cs['r'] = params[1]
    cs['a'] = params[2]

    sol = jump.solve(cs)
    v = sol.y[0,-1]
    return v

K = np.arange(10,111,10)
R = np.arange(0.02,0.121,0.01)
A = np.arange(0.6,1.41,0.1)
K, R, A = np.meshgrid(K,R,A)

KRA = np.array([K.ravel(),R.ravel(),A.ravel()]).T

if __name__ == '__main__':
    V = []
    p = multiprocessing.Pool()
    for i,v in enumerate(p.imap(sim,KRA,4)):
        V.append(v)
        print('{:d}/{:d}'.format(i+1,KRA.shape[0]),end='\r')

    V = np.array(V).reshape((-1,1))
    res = np.concatenate((KRA,V),axis=1)

    file_name = '../data/vrka.csv'
    data.write(
        file_name,
        ['k','r','a','v']+list(cs.keys()),
        res.T.tolist()+[[v] for v in list(cs.values())]
    )
