import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from anchor import design
from utils import plot

plot.set_default()

fig,ax = plt.subplots(
    1,1,
    figsize=(3.4-plot.pad*2,1.4-plot.pad*2),dpi=150
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
    c='C4'
    ax.plot(ka[:,0],ka[:,1],ls[i],color=c,lw=1,label=kas_label[i],markersize=5)
    ax.fill(ka[:,0],ka[:,1],color=c,ec=None,fill=True,alpha=0.4)

# ax.set_aspect(0.1/1.5) # goal as a square
xmargin = 0.2*0.12
ax.set_xlim(0.1-xmargin,0.2+xmargin)
ax.set_yscale('log')
ax.set_xticks(np.arange(0.1,0.21,0.05))
ax.set_yticks(np.arange(0.5,2.1,0.5),minor=False)
ax.set_yticks(np.arange(0.5,2.1,0.1),[],minor=True)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))
ax.set_xlabel('Stiffness Coefficient (Nm/rad)',labelpad=1)
ax.set_ylabel('Nonlinearity',labelpad=1)
ax.legend(loc='center left',handlelength=1,handletextpad=0.5,bbox_to_anchor=(1,0.5))

plt.subplots_adjust(
    left=0.11,right=0.815,top=1,bottom=0.23,
    wspace=0,hspace=0
)
plot.savefig('design.pdf',fig)
plt.show()
