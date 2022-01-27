import matplotlib.pyplot as plt
import jump
import numpy as np

cs = {
    'g': 9.81,
    'mb': 0.03,
    'ml': 0.0001,
    'k': 50,
    'a': 1,
    'ds': 0.05,
    'tau': 0.109872,
    'v': 30.410616886749196,
    'dl': 0.06,
    'r': 0.05
}
x0 = [0,0,0,0]

K = np.arange(20,110,10)
R = np.arange(0.02,0.11,0.01)
A = np.arange(0.5,1.6,0.25)
K, R, A = np.meshgrid(K,R,A)

V = []
for k, r, a in zip(K.flatten(),R.flatten(),A.flatten()):
    cs['k'] = k
    cs['r'] = r
    cs['a'] = a

    sol = jump.solve(x0, cs)
    v = sol.y[2,-1]
    V.append(v)
    print('k={:d} r={:.2f} a={:.2f}'.format(k,r,a))

V = np.array(V).reshape(K.shape)
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
print('Max k={:d} r={:.2f} a={:.2f} v={:.2f}'.format(k_max,r_max,a_max,v_max))
print('Min k={:d} r={:.2f} a={:.2f} v={:.2f}'.format(k_min,r_min,a_min,v_min))


fig, axes = plt.subplots(2,3,sharex=True,sharey=True)
for i,ax in enumerate(axes.ravel()):
    if i >= K.shape[2]:
        ax.axis('off')
        continue

    contour = ax.contourf(K[:,:,i],R[:,:,i],V[:,:,i],np.linspace(v_min,v_max,10))

    idx_max = np.argmax(V[:,:,i])
    k_max = K[:,:,i].flatten()[idx_max]
    r_max = R[:,:,i].flatten()[idx_max]
    v_max_a= V[:,:,i].flatten()[idx_max]

    ax.annotate(
        'vmax={:.2f}'.format(v_max_a),
        xy=(k_max, r_max),
        xytext=(15,15),textcoords='offset points',ha='center',va='bottom',
        arrowprops={'arrowstyle':'->'},
        bbox={'boxstyle':'round','fc':'w'}
    )
    ax.annotate(
        'a={:.2f}'.format(A[0,0,i]),
        xy=(0, 1),xycoords='axes fraction',
        xytext=(10,-10),textcoords='offset points',ha='left',va='top',
        bbox={'boxstyle':'round','fc':'w'}
    )

cb = fig.colorbar(contour,ax=axes,aspect=20,format=lambda x,p: '{:.2f}'.format(x))
cb.set_label('v [m/s]')

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('k [N/m]')
plt.ylabel('r [m]',labelpad=8)

plt.show()
