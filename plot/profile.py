import numpy as np
import matplotlib.pyplot as plt
from template import jump
from utils import plot

plot.set_default()
cs = jump.cs

fig,ax = plt.subplots(
    1,1,figsize=(3.25-plot.pad*2,1.25-plot.pad*2),dpi=150
)
lines = []
for k in [0.1,0.2]:
    for i,a in enumerate([0.5,1,2]):
        theta_d = np.linspace(0,cs['t']/k,100)
        theta_dp = np.zeros((7,theta_d.shape[0]))
        theta_dp[4,:] = theta_d
        cs = jump.cs
        cs['k'] = k
        cs['a'] = a
        tau_d = jump.f_ts(theta_dp,cs)

        c = 'C{:d}'.format(i)
        ls = '-' if k == 0.1 else '--'
        line = ax.plot(theta_d,tau_d,ls,color=c,linewidth=1)
        lines.append(line[0])


l1 = plt.legend(lines[:3],['$a=0.5$','$a=1.0$','$a=2.0$'],loc='upper left',handlelength=1,handletextpad=0.5,borderpad=0,labelspacing=0.5,borderaxespad=0,frameon=False,bbox_to_anchor=(0.04,1))
l2 = plt.legend(lines[::3],['$k=0.1\,Nm/rad$','$k=0.2\,Nm/rad$'],loc='lower right',handlelength=1,handletextpad=0.5,borderpad=0,labelspacing=0.5,borderaxespad=0,frameon=False,bbox_to_anchor=(1,0.08))
ax.add_artist(l1)
ax.add_artist(l2)
ax.spines[["left", "bottom"]].set_linewidth(1)
ax.spines[["left", "bottom"]].set_position(("data", 0))
ax.spines[["top", "right"]].set_visible(False)
ax.plot(1,0,'>k',markersize=3,transform=ax.get_yaxis_transform(),clip_on=False)
ax.plot(0,1,'^k',markersize=3,transform=ax.get_xaxis_transform(),clip_on=False)
ax.set_ylabel('$F$',labelpad=-3)
ax.set_xlabel('$x$',labelpad=-3)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-0.01,0.61])
ax.set_ylim([-0.0035,0.0635])

plt.subplots_adjust(
    left=0.025,right=1-0.015,top=1-0.04,bottom=0.02,
    wspace=0,hspace=0
)
plot.savefig('profile.pdf',fig)
plt.show()
