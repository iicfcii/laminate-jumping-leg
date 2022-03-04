import matplotlib.pyplot as plt
import jump
import numpy as np

cs = jump.cs
cs['k'] = 50
cs['r'] = 0.07

plt.figure()
for a in [0.6,1,1.4]:
    cs['a'] = a
    sol = jump.solve(cs)

    plt.plot(sol.t,sol.y[0,:],label='a={:.1f}'.format(a))
plt.legend()
plt.show()
