import matplotlib.pyplot as plt
import jump
import numpy as np

cs = {
    'g': 9.81,
    'mb': 0.03,
    'ml': 0.0001,
    'k': 40,
    'a': 0.5,
    'ds': 0.05,
    'tau': 0.109872,
    'v': 30.410616886749196,
    'dl': 0.06,
    'r': 0.05
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
