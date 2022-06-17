import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from utils import data
import anchor.jump as ajump
import template.jump as tjump
from stiffness import spring
from utils import plot
from anchor import design

plot.set_default()

fig,axes = plt.subplots(
    3,2,sharex=True,sharey='row',
    figsize=(3.4-plot.pad*2,4.8-plot.pad*2),dpi=150
)
lines = []
for i,s in enumerate(design.springs):
    k = s['k']
    a = s['a']

    # exp
    t,grf,dy,y,m,h = ajump.readn(1,k,a)
    p = grf*dy

    m = 0.028
    cs = tjump.cs
    cs['r'] = 0.04
    cs['mb'] = 0.02
    cs['mf'] =m-cs['mb']
    cs['k'] = s['kp']
    cs['a'] = s['ap']

    # sim
    cs['bs'] = 0
    sol = tjump.solve(cs)
    t_s = sol.t
    dy_s = sol.y[1,:]
    grf_s = tjump.f_grf(sol.y,cs)
    p_s = grf_s*dy_s

    # sim with damping
    cs['bs'] = s['b']
    sol = tjump.solve(cs)
    t_sb = sol.t
    dy_sb = np.interp(t_s,t_sb,sol.y[1,:])
    grf_sb = np.interp(t_s,t_sb,tjump.f_grf(sol.y,cs))
    p_sb = grf_sb*dy_sb

    lw = 0.8
    alpha = 0.5
    idx_a = 2-int(i/2)
    idx_k = i%2
    c = 'C{:d}'.format(idx_a)
    lines.append(axes[0,idx_k].plot(t_s,dy_s,'--',color=c,linewidth=lw)[0])
    lines.append(axes[0,idx_k].fill_between(t_s,dy_s,dy_sb,color=c,alpha=alpha,edgecolor=None))
    lines.append(axes[0,idx_k].plot(t,dy,color=c,linewidth=lw)[0])
    lines.append(axes[1,idx_k].fill_between(t_s,grf_s,grf_sb,color=c,alpha=alpha,edgecolor=None))
    lines.append(axes[1,idx_k].plot(t_s,grf_s,'--',color=c,linewidth=lw)[0])
    lines.append(axes[1,idx_k].plot(t,grf,color=c,linewidth=lw)[0])
    lines.append(axes[2,idx_k].fill_between(t_s,p_s,p_sb,color=c,alpha=alpha,edgecolor=None))
    lines.append(axes[2,idx_k].plot(t_s,p_s,'--',color=c,linewidth=lw)[0])
    lines.append(axes[2,idx_k].plot(t,p,color=c,linewidth=lw)[0])

xlim = axes[0,0].get_xlim()
axes[0,0].set_xlim(xlim[0],xlim[1]*1.04)
x_ticks = np.arange(0,0.081,0.02)
x_ticks_labels = ['{:.0f}'.format(t*100) for t in x_ticks]
axes[0,0].set_xticks(x_ticks,x_ticks_labels)
axes[0,0].set_ylabel('Body Speed (m/s)',labelpad=1)
axes[1,0].set_ylabel('GRF (N)',labelpad=1)
axes[2,0].set_ylabel('Power (W)',labelpad=1)
axes[2,0].set_xlabel('Time (10ms)',labelpad=1)
axes[2,1].set_xlabel('Time (10ms)',labelpad=1)
axes[0,0].legend(lines[36:],['Model','Damped','Experiment'],loc='upper left',handlelength=1,handletextpad=0.5)
axes[0,1].legend(lines[-1::-18],['0.5','1.0','2.0'],loc='upper left',handlelength=1,handletextpad=0.5)
axes[0,0].annotate(
    '0.1',
    xy=(1, 0),xycoords='axes fraction',
    xytext=(-2,2),textcoords='offset points',ha='right',va='bottom',
)
axes[0,1].annotate(
    '0.2',
    xy=(1, 0),xycoords='axes fraction',
    xytext=(-2,2),textcoords='offset points',ha='right',va='bottom',
)

plt.subplots_adjust(
    left=0.11,right=1,top=1,bottom=0.07,
    wspace=0.1,hspace=0
)
plot.savefig('jump.pdf',fig)
plt.show()
