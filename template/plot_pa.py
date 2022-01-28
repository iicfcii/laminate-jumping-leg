import matplotlib.pyplot as plt
import jump
import numpy as np

cs = {
    'g': 9.81,
    'mb': 0.03,
    'ml': 0.0001,
    'k': 30,
    'a': 1,
    'ds': 0.05, # max leg extension
    'tau': 0.109872,
    'v': 30.410616886749196,
    'dl': 0.06,
    'r': 0.05
}
x0 = [0,0,0,0]

A = [0.3,1]
V = []
lines = []
for i, a in enumerate(A):
    cs['a'] = a
    sol = jump.solve(x0, cs)

    vmax = sol.y[2,-1]
    f_spring = -np.sign(sol.y[1,:])*cs['k']*cs['ds']*np.power(np.abs(sol.y[1,:]/cs['ds']),cs['a'])
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
    [lines[i] for i in np.arange(2,2+len(A)*3,3)],
    ['a={:.1f} vmax={:.2f}'.format(a,v) for a,v in zip(A,V)],
    loc='upper right'
)

plt.xlabel('Time [s]')
plt.ylabel('Power [W]')
plt.show()
