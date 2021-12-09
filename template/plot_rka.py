import matplotlib.pyplot as plt
import jump
import numpy as np

pi = np.pi
cs = {
    'g': 9.81,
    'mb': 0.05,
    'ml': 0.01,
    'k': 1000,
    'a': 1,
    'el': 0.1, # max leg extension
    'tau': 0.215,
    'v': 383/60*2*pi,
    'em': 0.3,
    'r': 0.04
}
x0 = [0,0,0,0]
dim_sp = (3,3)

K = np.arange(100,1100,200)
A = np.arange(0.5,1.6,0.2)
K, A = np.meshgrid(K,A)
R = np.linspace(0.02,0.18,dim_sp[0]*dim_sp[1])

fig, axes = plt.subplots(*dim_sp,sharex=True,sharey=True)

for i,r in enumerate(R):
    cs['r'] = r

    V = []
    for k, a in zip(K.flatten(),A.flatten()):
            cs['k'] = k
            cs['a'] = a

            sol = jump.solve(x0, cs)
            v = sol.y[2,-1]
            V.append(v)
    V = np.array(V).reshape(K.shape)
    idx = np.argmax(V)
    k_opt = K.flatten()[idx]
    a_opt = A.flatten()[idx]

    plt.subplot(*dim_sp,i+1)
    vmin = 0
    vmax = 2
    plt.contourf(K,A,V,np.linspace(vmin,vmax,9),vmin=vmin,vmax=vmax)
    plt.annotate(
        'r={:.2f}'.format(cs['r'],np.amax(V)),
        xy=(1, 1),xycoords='axes fraction',
        xytext=(-10,-10),textcoords='offset points',ha='right',va='top',
        bbox={'boxstyle':'round','fc':'w'}
    )

    if np.unravel_index(idx,V.shape)[1] <= 1:
        ha='left'
        xt = 10
    else:
        ha='right'
        xt = -10
    plt.annotate(
        'vmax={:.2f}'.format(np.amax(V)),
        xy=(k_opt, a_opt),
        xytext=(xt,10),textcoords='offset points',ha=ha,va='bottom',
        arrowprops={'arrowstyle':'->'},
        bbox={'boxstyle':'round','fc':'w'}
    )

    if i == 1: plt.title('Launch speed, v [m/s] ({:.2}Nm, {:.0f}RPM)'.format(cs['tau'],cs['v']/2/pi*60))
    if i == 3: plt.ylabel('Linearity, a')
    if i == 7: plt.xlabel('Stiffness, k [N/m]')

    print('r = {:.2f} vmax = {:.2f}'.format(r,np.amax(V)))

plt.colorbar(ax=axes.ravel().tolist())
plt.show()
