import matplotlib.pyplot as plt
import jump
import numpy as np

cs = {
    'g': 9.81,
    'mb': 0.03,
    'ml': 0.0001,
    'k': 40,
    'a': 1,
    'ds': 0.05, # max leg extension
    'tau': 0.109872,
    'v': 30.410616886749196,
    'dl': 0.06,
    'r': 0.05
}
x0 = [0,0,0,0]

ys = np.linspace(-cs['ds'],cs['ds'],100)
A = [0.5,1,1.5]
lines = []
for i, a in enumerate(A):
    cs['a'] = a
    f_spring = np.sign(ys)*cs['k']*cs['ds']*np.power(np.abs(ys/cs['ds']),cs['a'])
    plt.plot(ys,f_spring,label='a={:.1f}'.format(a))
plt.legend()
plt.xlabel('y [m]')
plt.ylabel('F [N]')
plt.show()
