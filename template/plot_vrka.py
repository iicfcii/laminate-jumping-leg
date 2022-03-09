import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from utils import data
from utils import plot
from . import jump

f = data.read('./data/vrka.csv')
R = np.array(f['r'])
K = np.array(f['k'])
A = np.array(f['a'])
V = np.array(f['v'])

rka_shape = [len(np.unique(R)),len(np.unique(K)),len(np.unique(A))]
R = R.reshape(rka_shape)
K = K.reshape(rka_shape)
A = A.reshape(rka_shape)
V = V.reshape(rka_shape)

idx_max = np.argmax(V)
k_max = K.flatten()[idx_max]
r_max = R.flatten()[idx_max]
a_max = A.flatten()[idx_max]
v_max = V.flatten()[idx_max]

idx_min = np.argmin(V)
k_min = K.flatten()[idx_min]
r_min = R.flatten()[idx_min]
a_min = A.flatten()[idx_min]
v_min = V.flatten()[idx_min]
print('Max k={:.0f} r={:.2f} a={:.2f} v={:.2f}'.format(k_max,r_max,a_max,v_max))
print('Min k={:.0f} r={:.2f} a={:.2f} v={:.2f}'.format(k_min,r_min,a_min,v_min))

r_plots = [0.04,0.05,0.07,0.08,0.06]
# r_plots = R[:,0,0]

plot.set_default()

fig = plt.figure(figsize=(3.4-plot.pad*2,3.4),dpi=150)
gs = gridspec.GridSpec(2,1,figure=fig,height_ratios=[0.3,1])
gs1 = gs[0].subgridspec(1,4)
gs2 = gs[1].subgridspec(1,3,width_ratios=[0.13,1,0.28])

axes1 = [fig.add_subplot(gs1[0,i]) for i in range(4)]
axes2 = [fig.add_subplot(gs2[0,i]) for i in [1,0,2]]
axes = axes1+axes2

for j,ax in enumerate(axes[:-2]):
    # Figure out the r coordinate
    i = (np.abs(R[:,0,0].flatten()-r_plots[j])<1e-10).nonzero()[0][0]
    contour = ax.contourf(
        K[i,:,:],A[i,:,:],V[i,:,:],
        np.linspace(np.maximum(v_min,0),v_max,8),
        extend='neither' if v_min > 0 else 'min'
    )
    for c in contour.collections: c.set_edgecolor("face")
    ax.tick_params(axis='both',which='both')

    idx_max = np.argmax(V[i,:,:])
    k_max = K[i,:,:].flatten()[idx_max]
    a_max = A[i,:,:].flatten()[idx_max]
    v_max_r = V[i,:,:].flatten()[idx_max]

    ax.annotate(
        '{:.2f} m'.format(R[i,0,0]),
        xy=(1, 0),xycoords='axes fraction',
        xytext=(-5,5),textcoords='offset points',ha='right',va='bottom',
        bbox={'boxstyle':'round','fc':'w'}
    )

    if j == len(axes)-3:
        ax.annotate(
            '{:.2f}'.format(v_max_r),
            xy=(k_max, a_max),
            xytext=(15,-15),textcoords='offset points',ha='center',va='top',
            arrowprops={'arrowstyle':'->'},
            bbox={'boxstyle':'round','fc':'w'}
        )
        ax.set_ylabel('Nonlinearity',labelpad=2)
        ax.set_xlabel('Stiffness Coefficient (N/m)',labelpad=2)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

axes[-2].axis('off')
axes[-1].axis('off')
cb = fig.colorbar(
    contour,
    ax=axes[-1],
    format=lambda x,p: '{:.1f}'.format(x),
    location='right',
    aspect=10,
    pad=0,
    fraction=1
)
cb.set_label('Liftoff Velocity (m/s)')
cb.solids.set_edgecolor("face")

fig.subplots_adjust(left=0,right=1,top=1,bottom=0.1,wspace=0.1,hspace=0.05)
plot.savefig('sweep.pdf',fig)
plt.show()
