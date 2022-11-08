import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from anchor import design
from anchor import geom
from anchor.opt_spring import mass
from template import jump
from utils import plot

plot.set_default()

with PdfPages('spring.pdf') as pdf:
    lines = []
    for i in [4,2,0,5,3,1]:
        fig,ax = plt.subplots(1,1,figsize=(0.8-plot.pad*2,0.8-plot.pad*2),dpi=150)

        s = design.springs[i]
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

        ang = -geom.pose(lks[0,0,:])[1]
        tf = np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
        lks[0] = (tf@lks[0].transpose((0,2,1))).transpose((0,2,1))

        ang = ang-jump.cs['t']/k
        tf = np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
        lks[1] = (tf@lks[1].transpose((0,2,1))).transpose((0,2,1))

        bbox = geom.bbox(lks,pad=0)
        xc = (bbox[0]+bbox[1])/2
        yc = (bbox[2]+bbox[3])/2
        r = 0.042
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
            for n,link in enumerate(lk):
                lw = 2 if n==0 or n == 3 and j == 0 else 1
                line = ax.plot(
                    link[:,0],link[:,1],ls,color=c,
                    linewidth=lw,markersize=3
                )[0]
                if n == 0: lines.append(line)
                if n == 1: lines.append(line)

        if i == 1:
            # Scale
            l = 0.01
            x1 = xc+r-0.015-l/2
            x2 = x1+l
            y1 = yc-r+0.012
            y2 = y1
            ax.plot([x1,x2],[y1,y2],'k',linewidth=1)
            ax.annotate(
                '{:.0f}mm'.format(l*1000),
                xy=((x1+x2)/2, (y1+y2)/2),
                xytext=(0,-2),textcoords='offset points',ha='center',va='top',
            )

        plt.subplots_adjust(
            left=0,right=1,top=1,bottom=0,
            wspace=0,hspace=0
        )
        plot.savefig('spring_{:.0f}_{:.0f}.pdf'.format(k*100,a*10),fig,pdf=pdf)

    fig,ax = plt.subplots(1,1,figsize=(4.6-plot.pad*2,0.13-plot.pad*2),dpi=150)
    plt.subplots_adjust(
        left=0,right=1,top=1,bottom=0,
        wspace=0,hspace=0
    )
    ax.add_artist(plt.legend([lines[i] for i in [1,3,0]],['At Rest',r'Deformed to $\tau_{max}/k$','Input/Output Link'],loc='center left',ncol=3,handlelength=1.5,handletextpad=0.25,columnspacing=0.5,borderpad=0,borderaxespad=0,frameon=False))
    ax.add_artist(plt.legend([lines[i] for i in [1,5,9]],['$a=0.5$','$a=1.0$','$a=2.0$'],loc='center right',ncol=3,handlelength=1.5,handletextpad=0.25,columnspacing=0.5,borderpad=0,borderaxespad=0,frameon=False))
    ax.axis('off')
    plot.savefig('spring_legend.pdf',fig,pdf=pdf)
plt.show()
