import sys
sys.path.append('../template')

import numpy as np
import matplotlib.pyplot as plt
import fourbar
import jump
import opt

plt.figure('shape',figsize=(7.95,8))
for i,s in enumerate(opt.springs):
    cs = opt.cs
    cs['k'] = s['k']
    cs['a'] = s['a']
    x = s['x']
    ls = x[:4]
    w = x[4]
    c = x[5]

    lk = fourbar.spring(ls,c)
    color = 'C{:d}'.format(i+1)

    plt.subplot(3,3,i+1)
    lines = []
    line_styles = [':','-','--','-']
    line_widths = [2,3,2,2]
    for j,link in enumerate(lk):
        lines.append(plt.plot(link[:,0],link[:,1],line_styles[j],color=color,linewidth=line_widths[j])[0])
    # if i == 0: plt.legend(lines[:3],['input','ground','flexure'])

    plt.axis('scaled')
    ps = np.array(lk).reshape((-1,2))
    cx = (np.amax(ps[:,0])+np.amin(ps[:,0]))/2
    cy = (np.amax(ps[:,1])+np.amin(ps[:,1]))/2
    r = 0.05
    plt.xlim([cx-r,cx+r])
    plt.ylim([cy-r,cy+r])

    plt.xticks([])
    plt.yticks([])

    if i > 5: plt.xlabel('{:d}'.format(s['k']))
    if i % 3 == 0: plt.ylabel('{:.1f}'.format(s['a']),rotation='horizontal',ha='right')

plt.gcf().add_subplot(111, frameon=False)
plt.xticks([])
plt.yticks([])
plt.xlabel('k [N/m]',labelpad=16)
plt.ylabel('a',labelpad=20)
plt.subplots_adjust(wspace=0,hspace=0)
plt.show()
