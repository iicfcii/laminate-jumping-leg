import matplotlib.pyplot as plt
import numpy as np
from . import jump

cs = jump.cs
cs['k'] = 70
cs['r'] = 0.06
cs['ds'] = 0.06

for i,a in enumerate([0.7,1,1.5]):
    cs['a'] = a
    sol = jump.solve(cs,plot=False)
    v_max = np.amax(sol.y[0,:])
    ys_max = -np.amin(sol.y[1,:])
    print(ys_max,ys_max/cs['r'])

    ys = np.linspace(0,np.maximum(cs['ds'],ys_max),100)

    idx = np.nonzero(ys > ys_max)[0]
    idx = idx[0] if len(idx) > 0 else len(ys)
    ys1 = ys[:idx]
    ys2 = ys[idx-1:]
    f1 = -jump.f_spring(ys1,cs['k'],cs['a'],cs['ds'])
    f2 = -jump.f_spring(ys2,cs['k'],cs['a'],cs['ds'])

    c = 'C{:d}'.format(i)
    plt.plot(ys1,f1,color=c,label='a={:.1f} v_max={:.1f}'.format(a,v_max))
    plt.plot(ys2,f2,'--',color=c)
plt.legend()
plt.show()
