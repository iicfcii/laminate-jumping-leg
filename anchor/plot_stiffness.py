import numpy as np
import matplotlib.pyplot as plt
from . import design
from . import geom
from . import stiffness
from . import motion
from stiffness import leg
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
    x = s['x']

    thetad = np.linspace(0,cs['t']/cs['k'],100)
    taud = jump.t_spring(thetad,cs['k'],cs['a'],cs['t'])

    theta,tau = stiffness.sim(x,cs['t']/cs['k'],plot=False)

    # ye,fe = leg.readn(1,cs['k'],cs['a'])
    # kp,ap = leg.fit(ye,fe,cs['k'],cs['a'])
    # yf = ye
    # ff = -jump.f_spring(yf,kp,ap,cs['ds'])
    # print(cs['k'],cs['a'],kp,ap)

    ax.plot(thetad,taud,'--',color=c[i],linewidth=lw)
    ax.plot(theta,tau,'-',color=c[i],linewidth=lw)
    # ax.plot(ye,fe,'.',color=c[i],markersize=2)
    # ax.plot(yf,ff,'-.',color=c[i],linewidth=lw)
    # lines.append(ax.plot(yd,fd,'--',color=c[i],linewidth=lw)[0])
    # lines.append(ax.plot(y,f,'-',color=c[i],linewidth=lw)[0])
    # lines.append(ax.plot(y,f,'.',color=c[i],linewidth=lw,markersize=1.5)[0])
    # ax.set_ylim([-0.1,1.9])

# ax.legend(lines[:3],['Desired','Design','Measured'])
# ax.set_xlabel('Displacement (m)')
# ax.set_ylabel('Force (N)')
# plt.subplots_adjust(
#     left=0.14,right=1,top=1,bottom=0.125,
#     wspace=0,hspace=0
# )
# plot.savefig('stiffness.pdf',fig)
plt.show()
