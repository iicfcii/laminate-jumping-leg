import matplotlib.pyplot as plt
import jump
import numpy as np

cs = {
    'g': 9.81,
    'mb': 0.03,
    'ml': 0.0001,
    'k': 50,
    'a': 1,
    'el': 0.1, # max leg extension
    'tau': 0.109872,
    'v': 30.410616886749196,
    'em': 0.06,
    'r': 0.05
}
x0 = [0,0,0,0]

for r in [0.06]:
    cs['r'] = r
    plt.figure()
    for i, k in enumerate([50]):
        cs['k'] = k
        sol = jump.solve(x0, cs)

        vmax = sol.y[2,-1]
        f_spring = sol.y[1,:]*cs['k']
        dy_spring = sol.y[3,:]
        dy_body = sol.y[2,:]

        c = 'C{:d}'.format(i)
        plt.plot(sol.t,dy_spring*f_spring,'-.',color=c)
        plt.plot(sol.t,(dy_body-dy_spring)*f_spring,'--',color=c)
        plt.plot(sol.t,dy_body*f_spring,'-',color=c,label='k={:d} vmax={:.2f}'.format(k,vmax))

    plt.xlabel('Time [seconds]')
    plt.ylabel('Enery [W]')
    plt.legend()
    plt.title('-. spring, -- motor, - total, r={:.2f}'.format(r))

plt.show()
