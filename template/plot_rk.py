import matplotlib.pyplot as plt
import jump
import numpy as np

pi = np.pi
cs = {
    'g': 9.81,
    'mb': 0.05,
    'ml': 0.001,
    'k': 100,
    'a': 1,
    'el': 0.1, # max leg extension
    'tau': 0.215,
    'v': 383/60*2*pi,
    'em': 0.1,
    'r': 0.06
}
x0 = [0,0,0,0]

K = np.arange(100,600,100)
R = np.arange(0.02,0.16,0.02)
K, R = np.meshgrid(K,R)

V = []
for k, r in zip(K.flatten(),R.flatten()):
        cs['k'] = k
        cs['r'] = r

        sol = jump.solve(x0, cs)
        v = sol.y[2,-1]
        V.append(v)
V = np.array(V).reshape(K.shape)
idx = np.argmax(V)
k_opt = K.flatten()[idx]
r_opt = R.flatten()[idx]
v_opt = V.flatten()[idx]

print('k={:d} r={:.2f} v={:.2f}'.format(k_opt,r_opt,v_opt))

plt.show()
plt.contourf(K,R,V)

plt.annotate(
    'vmax={:.2f}'.format(v_opt),
    xy=(k_opt, r_opt),
    xytext=(10,10),textcoords='offset points',ha='left',va='bottom',
    arrowprops={'arrowstyle':'->'},
    bbox={'boxstyle':'round','fc':'w'}
)

plt.colorbar()
plt.ylabel('r [m]')
plt.xlabel('k [N/m]')
plt.show()
