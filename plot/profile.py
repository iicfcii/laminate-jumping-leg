import numpy as np
import matplotlib.pyplot as plt
from template import jump
from utils import plot

plot.set_default()
cs = jump.cs
k = 0.1

fig,ax = plt.subplots(
    1,1,figsize=(1.7-plot.pad*2,1.5-plot.pad*2),dpi=150
)
for i,a in enumerate([0.5,1,2]):
    theta_d = np.linspace(0,cs['t']/k,100)
    theta_dp = np.zeros((7,theta_d.shape[0]))
    theta_dp[4,:] = theta_d
    cs = jump.cs
    cs['k'] = k
    cs['a'] = a
    tau_d = jump.f_ts(theta_dp,cs)

    c = 'C{:d}'.format(i)
    ls = '-' if i == 1 else '--'
    ax.plot(theta_d,tau_d,ls,color=c,linewidth=1)

ax.legend(['$a=0.5$','$a=1.0$','$a=2.0$'],loc='upper left',handlelength=1,handletextpad=0.25,borderpad=0,labelspacing=0.25,borderaxespad=0,frameon=False,bbox_to_anchor=(0.08,1))
ax.spines[["left", "bottom"]].set_linewidth(1)
ax.spines[["left", "bottom"]].set_position(("data", 0))
ax.spines[["top", "right"]].set_visible(False)
ax.plot(1,0,'>k',markersize=3,transform=ax.get_yaxis_transform(),clip_on=False)
ax.plot(0,1,'^k',markersize=3,transform=ax.get_xaxis_transform(),clip_on=False)
ax.set_ylabel('$F$',labelpad=1)
ax.set_xlabel('$x$',labelpad=1)
ax.set_title('$k=0.1\,Nm/rad$',pad=4)

ax.set_xticks([])
ax.set_yticks([])

plt.subplots_adjust(
    left=0.05,right=1-0.05,top=1-0.12,bottom=0.03,
    wspace=0,hspace=0
)
plot.savefig('profile.pdf',fig)
plt.show()
