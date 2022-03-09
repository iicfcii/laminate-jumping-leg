import numpy as np
import matplotlib.pyplot as plt
from . import design
from . import geom
from template import jump
from utils import plot

plot.set_default()

cs = jump.cs
x = design.xm
ang = x[0]
l = x[1:6]
c = x[6]
rots = np.linspace(0,-cs['dl']/cs['r'],50)+x[0]
tilt = geom.leg(ang,l,c)[1]

lks = []
for rot in rots:
    lk, rot = geom.leg(rot,l,c,tilt=tilt)
    lks.append(lk)
lks = np.array(lks)
xs = lks[:,4,1,0]
ys = lks[:,4,1,1]

xs_d = np.zeros(xs.shape)
ys_d = (rots-x[0])*cs['r']+ys[0]

lksp = [lks[int(n)] for n in np.linspace(0,len(lks)-1,3)]
bbox = geom.bbox(lksp)

fig,axes = plt.subplots(
    1,2,
    figsize=(3.4-plot.pad*2,2.58),dpi=150
)

ax = axes[0]
ax.axis('scaled')
ax.set_xlim(bbox[:2])
ax.set_ylim(bbox[2:])
for i,lk in enumerate(lksp):
    # alpha = 0.3 if i > 0 else 1
    ls = '.--k' if i > 0 else '.-k'
    for link in lk:
        ax.plot(link[:,0],link[:,1],ls,linewidth=1,markersize=3)
pfs = lks[:,4,1,:]
ax.plot(pfs[:,0],pfs[:,1],'-r',linewidth=0.8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Fourbar Design')

axp = axes[1]
e_xs = (xs-xs_d)/cs['dl']*100
e_ys = (ys-ys_d)/cs['dl']*100
rots = rots[0]-rots
lw = 1
axp.plot(rots,e_xs,linewidth=lw,label='x')
axp.plot(rots,e_ys,linewidth=lw,label='y')
axp.legend()
axp.set_xlabel('Crank Angle (rad)')
axp.set_ylabel('Percentage Error (%)',labelpad=0)

plt.subplots_adjust(
    left=0,right=1,top=1,bottom=0.145,
    wspace=0.32,hspace=0
)
plot.savefig('fourbar.pdf',fig)
plt.show()
