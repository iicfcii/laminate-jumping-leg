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
bbox = geom.bbox(lksp,pad=0)
xc = (bbox[0]+bbox[1])/2
yc = (bbox[2]+bbox[3])/2

e_xs = (xs-xs_d)/cs['d']/cs['r']*100
e_ys = (ys-ys_d)/cs['d']/cs['r']*100

rs = motion.simf(rots,x)*1000
rsd = np.ones(len(rs))*40

w = 3.25
h = 1.5
wl = 0.92 # legend width
rx = 0.049
ry = h/((w-wl)/2)*rx
fig = plt.figure(figsize=(w-plot.pad*2,h-plot.pad*2),dpi=150)
gs_lk = fig.add_gridspec(nrows=1,ncols=2,left=wl/(w-plot.pad*2),bottom=0,right=1,top=1,wspace=0,hspace=0)
ax_lk = fig.add_subplot(gs_lk[0,0])
ax_pic = fig.add_subplot(gs_lk[0,1])
# gs_lk = fig.add_gridspec(nrows=1,ncols=1,left=0,bottom=0,right=0.4,top=1)
# gs_e = fig.add_gridspec(nrows=1,ncols=1,left=0.5,bottom=0.16,right=1,top=1)
# ax_lk = fig.add_subplot(gs_lk[0,0])
# ax_e = fig.add_subplot(gs_e[0,0])

ax_lk.axis('scaled')
ax_lk.set_xlim([xc-rx,xc+rx])
ax_lk.set_ylim([yc-ry,yc+ry])
ax_pic.axis('scaled')
ax_pic.set_xlim([xc-rx,xc+rx])
ax_pic.set_ylim([yc-ry,yc+ry])
lines = []
for i,lk in enumerate(lksp):
    ls = '.--k' if i > 0 else '.-k'
    for n,link in enumerate(lk):
        line = ax_lk.plot(link[:,0],link[:,1],ls,linewidth=1,markersize=3)[0]
        if n == 0: lines.append(line)
pfs = lks[:,4,1,:]
line = ax_lk.plot(pfs[:,0],pfs[:,1],'-',color='C0',linewidth=1)[0]
lines.append(line)

# Scale
l = 0.01
x1 = xc+rx-0.012-l/2
x2 = x1+l
y1 = yc-ry+0.011
y2 = y1
ax_lk.plot([x1,x2],[y1,y2],'k',linewidth=1)
ax_lk.annotate(
    '{:.0f}mm'.format(l*1000),
    xy=((x1+x2)/2, (y1+y2)/2),
    xytext=(0,-2),textcoords='offset points',ha='center',va='top',
)
l = ax_lk.legend(lines,['Retracted','Exteneded','Foot Trajectory'],loc='center right',ncol=1,handlelength=1.8,handletextpad=0.25,columnspacing=0.5,borderaxespad=0,borderpad=0,frameon=False,bbox_to_anchor=(0,0.5))
# bbox = l.get_window_extent(fig.canvas.get_renderer())
# print((bbox.x1-bbox.x0)/150,(bbox.y1-bbox.y0)/150)
# 0.9202777777777779 0.43111111111111117

ax_lk.set_xticks([])
ax_lk.set_yticks([])
ax_pic.set_xticks([])
ax_pic.set_yticks([])
ax_lk.axis('off')
ax_pic.axis('off')

plot.savefig('fourbar.pdf',fig)
plt.show()
