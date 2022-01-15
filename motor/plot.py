import sys
sys.path.append('../utils')

import matplotlib.pyplot as plt
import numpy as np
import spin
import data

pi = np.pi
cs = {
    'tau': 0.215*0.3,
    'v': 383/60*2*pi*0.97,
    # 'I': 0.04*0.06**2+8969/1e3/1e6
    'I': 0.04*0.09**2+25399/1e3/1e6
}
x0 = [0,0]
sol = spin.solve(x0,cs)

d = data.read('../data/motor_90_855.csv')
t = np.array(d['t'])
v = np.array(d['v'])*0.229/60*2*pi

plt.figure()
plt.plot(sol.t,sol.y[1,:])
plt.plot(t,v)
plt.show()
