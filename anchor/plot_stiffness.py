import numpy as np
import matplotlib.pyplot as plt
from . import design
from . import geom
from . import stiffness
from template import jump
from utils import plot

plot.set_default()

cs = jump.cs
fig,ax = plt.subplots(
    1,1,
    figsize=(3.4-plot.pad*2,3),dpi=150
)
lw = 0.8
c = ['C0','C1','C0','C1','C0','C1']
lines = []
for i,s in enumerate(design.springs):
    cs['k'] = s['k']
    cs['a'] = s['a']
    cs['r'] = 0.06

    sol = jump.solve(cs,plot=False)
    ys_max = -np.amin(sol.y[1,:])

    xd = np.linspace(0,cs['ds'],100)
    # xd = np.linspace(0,ys_max,100)
    fd = -jump.f_spring(xd,cs['k'],cs['a'],cs['ds'])


    rz,tz = stiffness.sim(s['x'],ys_max/cs['r'],plot=False)
    x = rz*cs['r']
    f = tz/cs['r']

    lines.append(ax.plot(xd,fd,'--',color=c[i],linewidth=lw)[0])
    lines.append(ax.plot(x,f,'-',color=c[i],linewidth=lw)[0])
    lines.append(ax.plot(x,f,'.',color=c[i],linewidth=lw,markersize=1.5)[0])
    ax.set_ylim([-0.1,1.9])

ax.legend(lines[:3],['Desired','Design','Measured'])
ax.set_xlabel('Displacement (m)')
ax.set_ylabel('Force (N)')
plt.subplots_adjust(
    left=0.14,right=1,top=1,bottom=0.125,
    wspace=0,hspace=0
)
plot.savefig('stiffness.pdf',fig)
plt.show()
