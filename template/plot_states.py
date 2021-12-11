import matplotlib.pyplot as plt
import jump
import numpy as np

pi = np.pi

cs = {
    'g': 9.81,
    'mb': 0.05,
    'ml': 0.001,
    'k': 100,
    'a': 1,
    'el': 0.1, # max leg extension
    'tau': 0.215,
    'v': 383/60*2*pi,
    'em': 0.1,
    'r': 0.06
}
x0 = [0,0,0,0]

sol = jump.solve(x0, cs)

plt.figure()
plt.subplot(221)
plt.plot(sol.t,sol.y[0,:])
plt.title('yb')
plt.subplot(223)
plt.plot(sol.t,sol.y[2,:])
plt.title('dyb')
plt.subplot(222)
plt.plot(sol.t,sol.y[1,:])
plt.title('yl')
plt.subplot(224)
plt.title('dyl')
plt.plot(sol.t,sol.y[3,:])
plt.show()
