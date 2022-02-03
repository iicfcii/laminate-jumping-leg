import matplotlib.pyplot as plt
import jump
import numpy as np

cs = {
    'g': 9.81,
    'mb': 0.035,
    'ml': 0.0001,
    'k': 30,
    'a': 0.5,
    'ds': 0.05,
    'tau': 0.109872,
    'v': 30.410616886749196,
    'dl': 0.06,
    'r': 0.05
}

plt.figure()
for a in [0.3]:
    cs['a'] = a
    sol = jump.solve(cs)
    plt.subplot(221)
    plt.plot(sol.t,sol.y[0,:],label='a={:.1f}'.format(a))
    plt.title('yb')
    plt.subplot(223)
    plt.plot(sol.t,sol.y[2,:])
    plt.title('dyb')
    plt.subplot(222)
    plt.plot(sol.t,sol.y[1,:])
    plt.title('ys')
    plt.subplot(224)
    plt.title('dys')
    plt.plot(sol.t,sol.y[3,:])
plt.subplot(221)
plt.legend()
plt.show()
