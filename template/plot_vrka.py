import matplotlib.pyplot as plt
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

r_plots = [0.06]
# r_plots = R[:,0,0]

plot.set_default()
fig,axes = plt.subplots(
    1,len(r_plots)+1,
    sharex=True,sharey=True,
    gridspec_kw={'width_ratios':len(r_plots)*[1]+[0.3]},
    figsize=(3.4, 2.5),dpi=150
)
# fig.patch.set_facecolor('red')
for j,ax in enumerate(axes.ravel()):
    if j >= len(r_plots):
        ax.axis('off')
        continue

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
        '{:.2f}'.format(v_max_r),
        xy=(k_max, a_max),
        xytext=(15,-15),textcoords='offset points',ha='center',va='top',
        arrowprops={'arrowstyle':'->'},
        bbox={'boxstyle':'round','fc':'w'}
    )
    ax.annotate(
        '{:.2f} m'.format(R[i,0,0]),
        xy=(0, 1),xycoords='axes fraction',
        xytext=(8,-8),textcoords='offset points',ha='left',va='top',
        bbox={'boxstyle':'round','fc':'w'}
    )
    # ax.set_title('r = {:.2f} m'.format(R[i,0,0]))

    ax.set_ylabel('Nonlinearity')
    ax.set_xlabel('Stiffness Coefficient (N/m)')

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

plt.subplots_adjust(left=0.14,right=1,top=0.95,bottom=0.15,wspace=0.1)
plt.show()
