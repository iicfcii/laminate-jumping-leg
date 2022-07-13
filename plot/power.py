import numpy as np
import matplotlib.pyplot as plt
from anchor import design
from utils import plot

plot.set_default()

fig,axes = plt.subplots(
    1,2,sharex=True,sharey='row',
    figsize=(3.4-plot.pad*2,1.7-plot.pad*2),dpi=150
)

ps = [[],[]]
for s in design.springs:
    i = 0 if s['k'] == 0.1 else 1
    ps[i].append([s['al'],s['p'],s['pb'],s['pe']])


lines = []
for i in range(2):
    p = np.array(ps[i])

    lines.append(axes[i].plot(p[:,0],p[:,1],'--*',lw=1,color='C4',markersize=5)[0])
    lines.append(axes[i].plot(p[:,0],p[:,2],'--',lw=1,color='C4')[0])
    # lines.append(axes[i].fill_between(p[:,0],p[:,1],p[:,2],color='C4',alpha=0.5,edgecolor=None))
    lines.append(axes[i].plot(p[:,0],p[:,3],'-',lw=1,color='C4')[0])

axes[1].legend(lines[:3],['No Damping','Model','Experiment'],loc='upper left',handlelength=1,handletextpad=0.5)

alla = np.array(ps)[:,:,0].ravel()
amin = np.amin(alla)
amax = np.amax(alla)
margin = (amax-amin)*0.1
axes[0].set_title('(a) $k=0.1\,Nm/rad$',pad=0,y=-0.37)
axes[1].set_title('(b) $k=0.2\,Nm/rad$',pad=0,y=-0.37)
axes[0].set_xlim(amin-margin,amax+margin)
axes[0].set_ylabel('Peak Power (W)',labelpad=1)
axes[0].set_xlabel('Nonlinearity',labelpad=1)
axes[1].set_xlabel('Nonlinearity',labelpad=1)

plt.subplots_adjust(
    left=0.11,right=1,top=1,bottom=0.29,
    wspace=0.1,hspace=0
)
plot.savefig('power.pdf',fig)
plt.show()
