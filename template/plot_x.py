import matplotlib.pyplot as plt
import jump
import numpy as np

cs = jump.cs
cs['k'] = 50
cs['r'] = 0.07

plt.figure()
for a in [0.5,1,1.5]:
    cs['a'] = a
    sol = jump.solve(cs)

    plt.subplot(411)
    plt.plot(sol.t,sol.y[0,:],label='a={:.1f}'.format(a))
    plt.subplot(412)
    plt.plot(sol.t,sol.y[1,:])
    plt.subplot(413)
    plt.plot(sol.t,sol.y[2,:])
    plt.subplot(414)
    plt.plot(sol.t,sol.y[3,:])

plt.subplot(411)
plt.legend()
plt.show()
