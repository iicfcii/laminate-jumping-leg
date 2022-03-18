import numpy as np
import matplotlib.pyplot as plt
from . import design
from . import geom
from template import jump
from utils import plot

plot.set_default()

cs = jump.cs
fig,axes = plt.subplots(
    3,2,
    figsize=(3.4-plot.pad*2,4.91),dpi=150
)
r = 0.04
scales = [2,1,1,0.7,0.7,0.7]
# scales = [1,1,1,1,1,1]
for i,ax in enumerate(axes.ravel()):
    s = design.springs[i]
    cs['k'] = s['k']
    cs['a'] = s['a']
    cs['r'] = 0.06

    sol = jump.solve(cs,plot=False)
    rot = -np.amin(sol.y[1,:])/cs['r']
    print(cs['k'],cs['a'],rot,rot*cs['r'])

    x = s['x']
    ls = x[:4]
    w = x[4]
    c = x[5]

    scale = scales[i]

    lks = []
    for ang in np.linspace(0,rot,2):
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
    if i%2 == 0: ax.set_ylabel('{:.1f}'.format(cs['a']))
    if i>3: ax.set_xlabel('{:d}'.format(cs['k']))

    for j,lk in enumerate(lks):
        ls = '.-k' if j == 0 else '.--k'
        for k,link in enumerate(lk):
            ax.plot(
                link[:,0],link[:,1],ls,
                linewidth=w/0.01*0.8 if k == 2 else 0.8,
                markersize=2
            )

    # Scale
    l = 0.01/scale
    x1 = xc+r-0.012-l/2
    x2 = x1+l
    y1 = yc+r-0.005
    y2 = y1
    ax.plot([x1,x2],[y1,y2],'k',linewidth=2)
    ax.annotate(
        '{:.2f} m'.format(l*scale),
        xy=((x1+x2)/2, (y1+y2)/2),
        xytext=(0,-5),textcoords='offset points',ha='center',va='top',
    )

ax = fig.add_subplot(111, frameon=False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Stiffness Coefficient (N/m)',labelpad=12)
ax.set_ylabel('Nonlinearity',labelpad=12)

plt.subplots_adjust(
    left=0.09,right=1,top=1,bottom=0.06,
    wspace=0,hspace=0
)
plot.savefig('springs.pdf',fig)
plt.show()
