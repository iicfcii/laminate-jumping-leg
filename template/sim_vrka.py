import sys
sys.path.append('../utils')

import multiprocessing
import matplotlib.pyplot as plt
import jump
import numpy as np
import data

cs = {
    'g': 9.81,
    'mb': 0.03,
    'ml': 0.0001,
    'k': 50,
    'a': 1,
    'ds': 0.05,
    'tau': 0.109872,
    'v': 30.410616886749196,
    'dl': 0.06,
    'r': 0.05
}

def sim(params):
    cs['k'] = params[0]
    cs['r'] = params[1]
    cs['a'] = params[2]

    sol = jump.solve(cs)
    v = sol.y[2,-1]
    return v

K = np.arange(10,110,10)
R = np.arange(0.02,0.11,0.01)
A = np.arange(0.2,1.71,0.1)
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
