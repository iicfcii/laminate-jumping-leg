import numpy as np
import matplotlib.pyplot as plt
from . import design
from . import geom
from . import stiffness
from template import jump
from utils import plot

plot.set_default()

cs = jump.cs
fig,axes = plt.subplots(
    1,2,
    figsize=(3.4,3),dpi=150
)

for i,s in enumerate(design.springs):
    cs['k'] = s['k']
    cs['a'] = s['a']
    cs['r'] = 0.06

    sol = jump.solve(cs,plot=False)

    ax = axes[0]
    ax.plot(sol.t,sol.y[0,:])
plt.show()
