import matplotlib.pyplot as plt
import jump
import numpy as np

cs = {
    'g': 9.81,
    'mb': 0.025,
    'ml': 0.0001,
    'k': 50,
    'a': 1,
    'ds': 0.05,
    'tau': 0.15085776558260747,
    'v': 47.75363911922214,
    'dl': 0.06,
    'r': 0.05
}

ys = np.linspace(0,cs['ds'],100)
A = [0.6,0.8,1,1.2,1.4]
lines = []
for i, a in enumerate(A):
    cs['a'] = a
    f = -jump.f_spring(ys,cs)
    plt.plot(ys,f,label='a={:.1f}'.format(a))
plt.legend()
plt.xlabel('Deformation')
plt.ylabel('Force')
plt.xticks([0,cs['ds']],['0','$\Delta_x$'])
plt.yticks([0,f[-1]],['0','$k\Delta_x$'])
plt.show()
