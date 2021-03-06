import matplotlib.pyplot as plt
import numpy as np
from . import jump

cs = jump.cs
cs['k'] = 0.1

for a in [0.7,1,1.5]:
    cs['a'] = a
    sol = jump.solve(cs)

    plt.figure(1)
    plt.plot(sol.t,sol.y[0,:],label='a={:.1f}'.format(a))
    plt.ylabel('y')
    plt.legend()
    plt.figure(2)
    plt.plot(sol.t,sol.y[1,:])
    plt.ylabel('dy')
    plt.figure(3)
    plt.plot(sol.t,sol.y[2,:])
    plt.ylabel('theta')
    plt.figure(4)
    plt.plot(sol.t,sol.y[3,:])
    plt.ylabel('dtheta')
    plt.figure(5)
    plt.plot(sol.t,sol.y[4,:])
    plt.ylabel('thetas')
    plt.figure(6)
    plt.plot(sol.t,sol.y[5,:])
    plt.ylabel('i')

    grf = jump.f_grf(sol.y,cs)
    plt.figure(7)
    plt.plot(sol.t,grf)
    plt.ylabel('grf')

plt.show()
