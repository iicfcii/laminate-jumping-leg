import numpy as np
import matplotlib.pyplot as plt
from anchor import design
from anchor import geom
from anchor.opt_spring import mass
from template import jump
from utils import plot

plot.set_default()

cs = jump.cs
fig,axes = plt.subplots(
    2,3,
    figsize=(3.4-plot.pad*2,2.3),dpi=150
)
r = 0.046
scales = [1,1,1,1,1,1]
axes = np.flip(axes.T)
for i,ax in enumerate(axes.ravel()):
    s = design.springs[i]
    cs['k'] = s['k']
    cs['a'] = s['a']
    x = s['x']

    print(cs['k'],cs['a'],cs['t']/cs['k'],mass(x))

    ls = x[:4]
    c = x[4]
    w = x[6]

    scale = scales[i]

    lks = []
    for ang in np.linspace(0,cs['t']/cs['k'],2):
        lk = geom.spring(ang,ls,c)
        lks.append(lk)
    lks = np.array(lks)/scale

    bbox = geom.bbox(lks,pad=0)
    xc = (bbox[0]+bbox[1])/2
    yc = (bbox[2]+bbox[3])/2

    ax.axis('scaled')
    ax.set_xlim([xc-r,xc+r])
    ax.set_ylim([yc-r,yc+r])
    ax.set_xticks([])
    ax.set_yticks([])
    if i%2 == 0: ax.set_xlabel('{:.1f}'.format(cs['a']),labelpad=2)
    if i>3: ax.set_ylabel('{:.2f}'.format(cs['k']),labelpad=0)

    for j,lk in enumerate(lks):
        ls = '.-k' if j == 0 else '.--k'
        for k,link in enumerate(lk):
            ax.plot(
                link[:,0],link[:,1],ls,
                linewidth=0.8,
                markersize=2
            )

    ax.annotate(
        '{:.0f} mm'.format(w*1000),
        xy=(1, 1),xycoords='axes fraction',
        xytext=(-2,-2),textcoords='offset points',ha='right',va='top',
    )

    if i == 0:
        # Scale
        l = 0.01/scale
        x1 = xc+r-0.015-l/2
        x2 = x1+l
        y1 = yc-r+0.015
        y2 = y1
        ax.plot([x1,x2],[y1,y2],'k',linewidth=2)
        ax.annotate(
            '{:.0f} mm'.format(l*scale*1000),
            xy=((x1+x2)/2, (y1+y2)/2),
            xytext=(0,-4),textcoords='offset points',ha='center',va='top',
        )

ax = fig.add_subplot(111, frameon=False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Nonlinearity',labelpad=10)
ax.set_ylabel('Stiffness Coefficient (N/m)',labelpad=10)

plt.subplots_adjust(
    left=0.09,right=1,top=1,bottom=0.11,
    wspace=0,hspace=0
)
plot.savefig('spring.pdf',fig)
plt.show()
