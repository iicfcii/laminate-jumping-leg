import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

e_xs = (xs-xs_d)/cs['d']/cs['r']*100
e_ys = (ys-ys_d)/cs['d']/cs['r']*100

rs = motion.simf(rots,x)*1000
rsd = np.ones(len(rs))*40

fig = plt.figure(figsize=(3.4-plot.pad*2,2.4-plot.pad*2),dpi=150)
gs_lk = fig.add_gridspec(nrows=1,ncols=1,left=0,bottom=0,right=0.4,top=1)
ax_lk = fig.add_subplot(gs_lk[0,0])
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.5,bottom=0.14,right=1,top=1)
ax_e = fig.add_subplot(gs[0,0])

ax_lk.axis('scaled')
ax_lk.set_xlim(bbox[:2])
ax_lk.set_ylim(bbox[2:])
for i,lk in enumerate(lksp):
    # alpha = 0.3 if i > 0 else 1
    ls = '.--k' if i > 0 else '.-k'
    for link in lk:
        ax_lk.plot(link[:,0],link[:,1],ls,linewidth=1,markersize=3)
pfs = lks[:,4,1,:]
ax_lk.plot(pfs[:,0],pfs[:,1],'.k',markersize=3,markevery=3)

# Scale
l = 0.01
x1 = bbox[1]-0.012-l/2
x2 = x1+l
y1 = bbox[2]+0.012
y2 = y1
ax_lk.plot([x1,x2],[y1,y2],'k',linewidth=2)
ax_lk.annotate(
    '{:.0f}mm'.format(l*1000),
    xy=((x1+x2)/2, (y1+y2)/2),
    xytext=(0,-4),textcoords='offset points',ha='center',va='top',
)

ax_lk.set_xticks([])
ax_lk.set_yticks([])
ax_lk.axis('off')

ax_e.plot(rots,e_xs,linewidth=1,label='x')
ax_e.plot(rots,e_ys,linewidth=1,label='y')
ax_e.legend()
ax_e.set_xlabel('Crank Angle (rad)',labelpad=1)
ax_e.set_ylabel('Percentage Error (%)',labelpad=1)

# ax_r.plot(rots,rsd,'--',linewidth=1,label='Goal')
# ax_r.plot(rots,rs,linewidth=1,label='Model')
# ax_r.legend()
# ax_r.set_yticks(np.linspace(25,45,5))
# ax_r.set_xlabel('Crank Angle (rad)',labelpad=1)
# ax_r.set_ylabel('Virtual Radius (mm)',labelpad=1)

plot.savefig('fourbar.pdf',fig)
plt.show()
