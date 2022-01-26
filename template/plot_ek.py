import matplotlib.pyplot as plt
import jump
import numpy as np

cs = {
    'g': 9.81,
    'mb': 0.03,
    'ml': 0.0001,
    'k': 50,
    'a': 1,
    'el': 0.05, # max leg extension
    'tau': 0.109872,
    'v': 30.410616886749196,
    'em': 0.06,
    'r': 0.05
}
x0 = [0,0,0,0]

K = [20,50,80]
V = []
lines = []
for i, k in enumerate(K):
    cs['k'] = k
    sol = jump.solve(x0, cs)

    vmax = sol.y[2,-1]
    # f_spring = -sol.y[1,:]*cs['k']
    f_spring = -np.sign(sol.y[1,:])*cs['k']*cs['el']*np.power(np.abs(sol.y[1,:]/cs['el']),cs['a'])
    dy_spring = sol.y[3,:]
    dy_body = sol.y[2,:]

    c = 'C{:d}'.format(i)
    ls, = plt.plot(sol.t,dy_spring*f_spring,'-.',color=c)
    lm, = plt.plot(sol.t,(dy_body-dy_spring)*f_spring,'--',color=c)
    lt, = plt.plot(sol.t,dy_body*f_spring,'-',color=c)

    V.append(vmax)
    lines.append(ls)
    lines.append(lm)
    lines.append(lt)

type_legend = plt.legend(
    lines[0:3],
    ['spring','motor','total'],
    loc='lower right',
)
plt.gca().add_artist(type_legend)
plt.legend(
    [lines[2],lines[5],lines[8]],
    ['k={:d} vmax={:.2f}'.format(k,v) for k,v in zip(K,V)],
    loc='upper right'
)

plt.xlabel('Time [seconds]')
plt.ylabel('Enery [W]')
plt.show()
