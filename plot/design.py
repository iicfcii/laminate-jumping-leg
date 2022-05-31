import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from anchor import design
from anchor import geom
from anchor import motion
from template import jump
from utils import plot

plot.set_default()

fig = plt.figure(figsize=(3.4-plot.pad*2,3.56),dpi=150)
gs = gridspec.GridSpec(1,2,figure=fig,width_ratios=[0.043*4,0.0735])
gs1 = gs[0].subgridspec(3,2)
gs2 = gs[1].subgridspec(1,1)

axes_s = [fig.add_subplot(gs1[r,c]) for r in range(3) for c in range(2)]
axes_f = [fig.add_subplot(gs2[0,0])]

cs = jump.cs
lw = 1
ms = 3


# Springs
r = 0.043
for i,ax in enumerate(axes_s):
    s = design.springs[i]
    cs['k'] = s['k']
    cs['a'] = s['a']
    x = s['x']

    ls = x[:4]
    c = x[4]
    w = x[6]

    lks = []
    for ang in np.linspace(0,cs['t']/cs['k'],2):
        lk = geom.spring(ang,ls,c)
        lks.append(lk)
    lks = np.array(lks)

    bbox = geom.bbox(lks,pad=0)
    xc = (bbox[0]+bbox[1])/2
    yc = (bbox[2]+bbox[3])/2

    ax.axis('scaled')
    ax.set_xlim([xc-r,xc+r])
    ax.set_ylim([yc-r,yc+r])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    idx_a = int(i/2)
    # idx_k = i%2
    c = 'C{:d}'.format(2-idx_a)
    for j,lk in enumerate(lks):
        ls = '.-' if j == 0 else '.--'
        for k,link in enumerate(lk):
            ax.plot(
                link[:,0],link[:,1],ls,color=c,
                linewidth=lw,markersize=ms
            )

    ax.annotate(
        '{:.1f}, {:.1f}, {:.0f}'.format(cs['k'],cs['a'],w*1000),
        xy=(0.5, 0),xycoords='axes fraction',
        xytext=(0,2),textcoords='offset points',ha='center',va='bottom',
    )

    # ax.annotate(
    #     '{:.1f}'.format(cs['a']),
    #     xy=(0, 1),xycoords='axes fraction',
    #     xytext=(2,-2),textcoords='offset points',ha='left',va='top',
    # )
    # ax.annotate(
    #     '{:.1f}Nm/rad'.format(cs['k']),
    #     xy=(1, 1),xycoords='axes fraction',
    #     xytext=(-2,-2),textcoords='offset points',ha='right',va='top',
    # )
    # ax.annotate(
    #     '{:.0f}mm'.format(w*1000),
    #     xy=(1, 0),xycoords='axes fraction',
    #     xytext=(-2,2),textcoords='offset points',ha='right',va='bottom',
    # )

# Fourbar
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

ax = axes_f[0]
ax.axis('scaled')
ax.set_xlim(bbox[:2])
ax.set_ylim(bbox[2:])
for i,lk in enumerate(lksp):
    # alpha = 0.3 if i > 0 else 1
    ls = '.--k' if i > 0 else '.-k'
    for link in lk:
        ax.plot(link[:,0],link[:,1],ls,linewidth=lw,markersize=ms)
pfs = lks[:,4,1,:]
ax.plot(pfs[:,0],pfs[:,1],'.k',markersize=ms,markevery=3)

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
ax.axis('off')

plt.subplots_adjust(
    left=0,right=1,top=1,bottom=0,
    wspace=0,hspace=0
)
plot.savefig('design.pdf',fig)
plt.show()
