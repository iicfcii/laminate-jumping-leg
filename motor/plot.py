import sys
sys.path.append('../utils')

import matplotlib.pyplot as plt
import numpy as np
import spin
import data

# Exp motor data
d = data.read('../data/pololu_hpcp100_90_6V_1.csv',skip=6)
t = np.array(d['Time (Seconds)'])
x = np.array(d['X'])
x1 = np.array(d['X1'])
z = np.array(d['Z'])
z1 = np.array(d['Z1'])
theta = np.arctan2(z1-z,x1-x)

dtheta = (theta[1:]-theta[:-1])/(t[1:]-t[:-1])
t = t[1:]

loop_idx = dtheta < -0.5
dtheta = dtheta[loop_idx]
t = t[loop_idx]
t -= t[0]

time_idx = t < 1
dtheta = dtheta[time_idx]
t = t[time_idx]

v = -dtheta

# Simulation
cs = {
    'tau': 1.6*9.81/100*0.7,
    'v': 330/60*2*np.pi*0.88,
    # 'I': 0.04*0.06**2+8969/1e3/1e6
    'I': 0.04*0.09**2+25399/1e3/1e6
}
x0 = [0,0]
sol = spin.solve(x0,cs)

plt.figure()
plt.plot(sol.t,sol.y[1,:])
plt.plot(t,v)
plt.show()
