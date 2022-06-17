import numpy as np
import matplotlib.pyplot as plt
from anchor import design
from utils import plot

plot.set_default()

fig,ax = plt.subplots(
    1,1,
    figsize=(3.4-plot.pad*2,2.35-plot.pad*2),dpi=150
)

kas = [[],[],[],[]]
kas_label =  ['Goal','Model','Spring','Leg']
for s in [design.springs[i] for i in [0,2,4,5,3,1]]:
    kas[0].append([s['k'],s['a']])
    kas[1].append([s['km'],s['am']])
    kas[2].append([s['ks'],s['as']])
    kas[3].append([s['kl'],s['al']])

ls = ['--','^-','.-','-']
for i,ka in enumerate(kas):
    ka = np.array(ka)
    ka = np.concatenate([ka,ka[[0],:]],axis=0)
    # c = 'C{:d}'.format(i)
    c='C4'
    # ax.fill(ka[:,0],ka[:,1],color=c,ec=c,lw=1,fill=False,label=kas_label[i])
    ax.plot(ka[:,0],ka[:,1],ls[i],color=c,lw=1,label=kas_label[i])
    ax.fill(ka[:,0],ka[:,1],color=c,ec=None,fill=True,alpha=0.4)

ax.set_aspect(0.1/1.5) # goal as a square
xmargin = 0.2*0.14
ax.set_xlim(0.1-xmargin,0.2+xmargin)
# ymargin = 1.5*0.1
# ax.set_ylim(0.5-ymargin,2+ymargin)
y_ticks = np.arange(0.5,2.01,0.25)
ax.set_yticks(y_ticks)
ax.set_xlabel('Stiffness Coefficient (Nm/rad)',labelpad=1)
ax.set_ylabel('Nonlinearity',labelpad=1)
ax.legend(loc='center left',handlelength=1,handletextpad=0.5)

plt.subplots_adjust(
    left=0.125,right=1,top=1,bottom=0.13,
    wspace=0.1,hspace=0
)
plot.savefig('designspace.pdf',fig)
plt.show()
