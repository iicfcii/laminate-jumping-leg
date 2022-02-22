import sys
sys.path.append('../utils')

import matplotlib.pyplot as plt
import jump
import numpy as np
import data

f = data.read('../data/vrka.csv')
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

a_plots = A[0,0,:].flatten()
fig, axes = plt.subplots(3,3,sharex=True,sharey=True)
for j,ax in enumerate(axes.ravel()):
    if j >= K.shape[2]:
        ax.axis('off')
        continue

    i = (np.abs(A[0,0,:].flatten()-a_plots[j])<1e-10).nonzero()[0][0]
    contour = ax.contourf(
        K[:,:,i],R[:,:,i],V[:,:,i],
        np.linspace(np.maximum(v_min,0),v_max,11),
        extend='neither' if v_min > 0 else 'min'
    )

    idx_max = np.argmax(V[:,:,i])
    k_max = K[:,:,i].flatten()[idx_max]
    r_max = R[:,:,i].flatten()[idx_max]
    v_max_a= V[:,:,i].flatten()[idx_max]

    ax.annotate(
        '{:.2f}'.format(v_max_a),
        xy=(k_max, r_max),
        xytext=(15,15),textcoords='offset points',ha='center',va='bottom',
        arrowprops={'arrowstyle':'->'},
        bbox={'boxstyle':'round','fc':'w'}
    )
    ax.annotate(
        'a={:.2f}'.format(A[0,0,i]),
        xy=(1, 0),xycoords='axes fraction',
        xytext=(-10,10),textcoords='offset points',ha='right',va='bottom',
        bbox={'boxstyle':'round','fc':'w'}
    )

cb = fig.colorbar(contour,ax=axes,aspect=20,format=lambda x,p: '{:.2f}'.format(x))
cb.set_label('v [m/s]')

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('k [N/m]')
plt.ylabel('r [m]',labelpad=8)

plt.show()
