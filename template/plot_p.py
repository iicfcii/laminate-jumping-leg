import matplotlib.pyplot as plt
import numpy as np
from . import jump

cs = jump.cs

for k in [0.2]:
    for i,a in enumerate([0.7,1,1.5]):
        cs['k'] = k
        cs['a'] = a
        sol = jump.solve(cs,plot=False)
        t = sol.t
        grf = jump.t_spring(sol.y[4,:],cs['k'],cs['a'],cs['t'])/cs['r']
        dy = sol.y[1,:]
        p = grf*dy

        pm = sol.y[5,:]*cs['K']*sol.y[3,:]
        pe = sol.y[5,:]*cs['V']

        v_max = np.amax(sol.y[1,:])

        c = 'C{:d}'.format(i)
        plt.figure(1)
        plt.plot(t,p,'-',color=c,label='a={:.1f} v_max={:.2f}'.format(a,v_max))
        plt.plot(t,pm,'--',color=c)
        plt.figure(2)
        plt.plot(t,p,'-',color=c,label='a={:.1f} v_max={:.2f}'.format(a,v_max))
        plt.plot(t,pe,'--',color=c)
plt.figure(1)
plt.legend()
plt.figure(2)
plt.legend()
plt.show()
