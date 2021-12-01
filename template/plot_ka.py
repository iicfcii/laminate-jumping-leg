import matplotlib.pyplot as plt
import jump
import numpy as np

pi = np.pi
cs = {
    'g': 9.81,
    'mb': 0.03,
    'ml': 0.01,
    'k': 1000,
    'a': 1,
    'el': 0.1, # max leg extension
    'tau': 0.215,
    'v': 383/60*2*pi,
    'em': pi,
    'r': 0.04
}
x0 = [0,0,0,0]

K = np.arange(100,1600,100)
A = np.arange(0.5,1.6,0.1)
K, A = np.meshgrid(K,A)
V = []
for k, a in zip(K.flatten(),A.flatten()):
        cs['k'] = k
        cs['a'] = a

        sol = jump.solve(x0, cs)
        v = sol.y[2,-1]
        V.append(v)
V = np.array(V).reshape(K.shape)
print(np.amax(V))

fig = plt.figure()
plt.contourf(K,A,V)
plt.colorbar()
# plt.xticks(K[0,:])
# plt.yticks(A[:,0])
plt.show()
