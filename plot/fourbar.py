import numpy as np
import matplotlib.pyplot as plt
from anchor import design
from anchor import geom
from anchor import motion
from template import jump
from utils import plot

plot.set_default()

cs = jump.cs
x = design.xm
ang = x[0]
l = x[1:6]
c = x[6]
rots = np.linspace(0,-cs['d'],50)+x[0]
tilt = geom.leg(ang,l,c,motion.tr)[1]

lks = []
for rot in rots:
    lk, rot = geom.leg(rot,l,c,motion.tr,tilt=tilt)
    lks.append(lk)
lks = np.array(lks)
xs = lks[:,4,1,0]
ys = lks[:,4,1,1]

xs_d = np.zeros(xs.shape)
ys_d = (rots-x[0])*cs['r']+ys[0]

lksp = [lks[int(n)] for n in np.linspace(0,len(lks)-1,2)]
bbox = geom.bbox(lksp)

fig,axes = plt.subplots(
    1,1,
    figsize=(3.4-plot.pad*2,2),dpi=150
)

ax = axes
ax.axis('scaled')
ax.set_xlim(bbox[:2])
ax.set_ylim(bbox[2:])
for i,lk in enumerate(lksp):
    # alpha = 0.3 if i > 0 else 1
    ls = '.--k' if i > 0 else '.-k'
    for link in lk:
        ax.plot(link[:,0],link[:,1],ls,linewidth=1,markersize=3)
pfs = lks[:,4,1,:]
ax.plot(pfs[:,0],pfs[:,1],'.k',markersize=3,markevery=3)

# Scale
l = 0.01
x1 = bbox[1]-0.012-l/2
x2 = x1+l
y1 = bbox[2]+0.012
y2 = y1
ax.plot([x1,x2],[y1,y2],'k',linewidth=2)
ax.annotate(
    '{:.0f}mm'.format(l*1000),
    xy=((x1+x2)/2, (y1+y2)/2),
    xytext=(0,-4),textcoords='offset points',ha='center',va='top',
)

ax.set_xticks([])
ax.set_yticks([])

# axp = axes[1]
# e_xs = (xs-xs_d)/cs['d']/cs['r']*100
# e_ys = (ys-ys_d)/cs['d']/cs['r']*100
# rots = rots[0]-rots
# lw = 1
# axp.plot(rots,e_xs,linewidth=lw,label='x')
# axp.plot(rots,e_ys,linewidth=lw,label='y')
# axp.legend()
# axp.set_xlabel('Crank Angle (rad)')
# axp.set_ylabel('Percentage Error (%)',labelpad=0)

plt.subplots_adjust(
    left=0,right=1,top=1,bottom=0,
    wspace=0,hspace=0
)
plot.savefig('fourbar.pdf',fig)
plt.show()
