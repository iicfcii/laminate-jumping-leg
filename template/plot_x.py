import matplotlib.pyplot as plt
import numpy as np
from . import jump

cs = jump.cs
cs['k'] = 70

plt.figure()
for a in [0.7,1,1.5]:
    cs['a'] = a
    sol = jump.solve(cs)

    plt.subplot(511)
    plt.plot(sol.t,sol.y[0,:],label='a={:.1f}'.format(a))
    plt.subplot(512)
    plt.plot(sol.t,sol.y[1,:])
    plt.subplot(513)
    plt.plot(sol.t,sol.y[2,:])
    plt.subplot(514)
    plt.plot(sol.t,sol.y[3,:])
    plt.subplot(515)
    plt.plot(sol.t,sol.y[4,:])

plt.subplot(511)
plt.legend()
plt.show()
