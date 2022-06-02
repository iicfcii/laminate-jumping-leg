import numpy as np
import matplotlib.pyplot as plt
from anchor import design
from anchor import geom
from anchor.opt_spring import mass
from template import jump
from utils import plot

plot.set_default()

r = 0.042
for i,s in enumerate(design.springs):
    fig,ax = plt.subplots(1,1,figsize=(0.9-plot.pad*2,0.9-plot.pad*2),dpi=150)

    k = s['k']
    a = s['a']
    x = s['x']
    ls = x[:4]
    c = x[4]
    w = x[6]

    print(k,a,w)

    lks = []
    for ang in np.linspace(0,jump.cs['t']/k,2):
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
        for link in lk:
            ax.plot(
                link[:,0],link[:,1],ls,color=c,
                linewidth=1,markersize=3
            )

    if i == 1:
        # Scale
        l = 0.01
        x1 = xc+r-0.015-l/2
        x2 = x1+l
        y1 = yc-r+0.015
        y2 = y1
        ax.plot([x1,x2],[y1,y2],'k',linewidth=2)
        ax.annotate(
            '{:.0f}mm'.format(l*1000),
            xy=((x1+x2)/2, (y1+y2)/2),
            xytext=(0,-4),textcoords='offset points',ha='center',va='top',
        )

    plt.subplots_adjust(
        left=0,right=1,top=1,bottom=0,
        wspace=0,hspace=0
    )
    plot.savefig('spring_{:.0f}_{:.0f}.pdf'.format(k*100,a*10),fig)
plt.show()
