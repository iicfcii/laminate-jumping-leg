import numpy as np
import matplotlib.pyplot as plt
from anchor import design
from anchor import geom
from anchor import stiffness
from anchor import motion
from stiffness import spring
from template import jump
from utils import plot

plot.set_default()

cs = jump.cs
fig,ax = plt.subplots(
    1,1,
    figsize=(3.4-plot.pad*2,2.7),dpi=150
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

    thetae,taue = spring.readn(0,cs['k'],cs['a'])
    idx = thetae < cs['t']/cs['k']+0.01
    thetae = thetae[idx]
    taue = taue[idx]

    kp,ap = spring.fit(thetae,taue,cs['k'],cs['a'])
    print(cs['k'],cs['a'],kp,ap)

    lines.append(ax.plot(thetad,taud,'--',color=c[i],linewidth=lw)[0])
    lines.append(ax.plot(theta,tau,'-',color=c[i],linewidth=lw)[0])
    lines.append(ax.plot(thetae,taue,'.-',color=c[i],linewidth=lw,markersize=2)[0])

ax.legend(lines[:3],['Target','Simulation','Experiment'],loc='lower right',handlelength=1,handletextpad=0.5)
ax.set_xlabel('Rotation (rad)',labelpad=1)
ax.set_ylabel('Torque (Nm)',labelpad=1)
plt.subplots_adjust(
    left=0.125,right=1,top=1,bottom=0.12,
    wspace=0,hspace=0
)
plot.savefig('stiffness.pdf',fig)
plt.show()
