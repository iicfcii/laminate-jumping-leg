import matplotlib.pyplot as plt
import jump
import numpy as np

cs = jump.cs
cs['k'] = 30
cs['r'] = 0.06

ys = np.linspace(0,cs['d'],100)
A = [0.5,1,1.5]
lines = []
for i, a in enumerate(A):
    cs['a'] = a
    f = -jump.f_spring(ys,cs['k'],cs['a'],cs['d'])
    plt.plot(ys,f,label='a={:.1f}'.format(a))
plt.legend()
plt.xlabel('Deformation')
plt.ylabel('Force')
plt.xticks([0,cs['d']],['0','d'])
plt.yticks([0,f[-1]],['0','k'])
plt.show()
