import matplotlib.pyplot as plt
import jump
import numpy as np

pi = np.pi

cs = {
    'g': 9.81,
    'mb': 0.03,
    'ml': 0.0001,
    'k': 70,
    'a': 1,
    'el': 0.1, # max leg extension
    'tau': 0.215*0.3,
    'v': 383/60*2*pi*0.97,
    'em': 0.05,
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
