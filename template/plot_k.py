import matplotlib.pyplot as plt
import numpy as np
from . import jump

cs = jump.cs
cs['k'] = 70*cs['r']**2

for i,a in enumerate([0.5,1,2]):
    cs['a'] = a
    sol = jump.solve(cs,plot=False)
    v_max = np.amax(sol.y[1,:])
    thetas_max = np.amax(sol.y[4,:])
    print(thetas_max)

    thetas = np.linspace(0,np.maximum(cs['t']/cs['k'],thetas_max),100)

    idx = np.nonzero(thetas > thetas_max)[0]
    idx = idx[0] if len(idx) > 0 else len(thetas)
    thetas1 = thetas[:idx]
    thetas2 = thetas[idx-1:]
    ts1 = jump.t_spring(thetas1,cs['k'],cs['a'],cs['t'])
    ts2 = jump.t_spring(thetas2,cs['k'],cs['a'],cs['t'])

    c = 'C{:d}'.format(i)
    plt.plot(thetas1,ts1,color=c,label='a={:.1f} v_max={:.2f}'.format(a,v_max))
    plt.plot(thetas2,ts2,'--',color=c)
plt.legend()
plt.show()
