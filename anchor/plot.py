import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from utils import data
from template import jump

sos = butter(2,50,'lowpass',fs=1000,output='sos')
sos_dy = butter(2,50,'lowpass',fs=360,output='sos')

def read(s,k,a,n):
    name = './data/jump_{:d}_{:d}_{:d}_{:d}.csv'.format(s,k,int(a*10),n)
    d = data.read(name)
    t = np.array(d['t'])
    y_raw = np.array(d['y'])
    grf_raw = np.array(d['grf'])

    t_y = t[y_raw != None].astype('float64')
    y_raw =  y_raw[y_raw != None].astype('float64')
    dy_raw = np.concatenate(([0],(y_raw[1:]-y_raw[:-1])/(t_y[1:]-t_y[:-1])))

    # Filter
    grf = sosfiltfilt(sos,grf_raw)
    dy = sosfiltfilt(sos_dy,dy_raw)

    # Find start and lift off time
    grf_bias = None
    ti = 0.5
    while ti < 1:
        idx = np.logical_and(t > ti,t < ti+0.1)
        std = np.std(grf[idx])
        if std < 0.001:
            grf_bias = np.average(grf[idx])
            break
        ti += 0.005
    assert grf_bias is not None, 'Cant find no load bias'

    ti = t[np.nonzero(t > 0.505)[0][0]]
    tf = t[np.nonzero(grf > grf_bias)[0][0]]
    # ti = 0.5
    # tf = 0.8

    # plt.plot(t,grf)
    # plt.plot(t,grf_raw)
    # plt.show()

    # Select
    idx_ti_grf = np.nonzero(t > ti)[0][0]
    idx_tf_grf = np.nonzero(t > tf)[0][0]
    t = t[idx_ti_grf:idx_tf_grf]
    grf_raw = -(grf_raw[idx_ti_grf:idx_tf_grf]-grf_bias)
    grf = -(grf[idx_ti_grf:idx_tf_grf]-grf_bias)

    idx_ti_y = np.nonzero(t_y > ti)[0][0]
    idx_tf_y = np.nonzero(t_y > tf)[0][0]
    t_y = t_y[idx_ti_y:idx_tf_y]
    dy_raw = dy_raw[idx_ti_y:idx_tf_y]
    dy = dy[idx_ti_y:idx_tf_y]

     # Upsample to make data simpler
    dy_raw = np.interp(t,t_y,dy_raw)
    dy = np.interp(t,t_y,dy)

    t -= t[0]

    # plt.figure()
    # plt.plot(t,grf_raw)
    # plt.plot(t,grf)

    # plt.figure()
    # plt.plot(t,dy_raw)
    # plt.plot(t,dy)
    # plt.show()

    return t,grf,dy

if __name__ == '__main__':
    plt.figure()
    for k in [70]:
        for i,a in enumerate([0.7,1,1.5]):
            for s in [1]:
                c = 'C{:d}'.format(i)

                ts = []
                grfs = []
                dys = []
                for n in [1,2,3]:
                    t,grf,dy = read(s,k,a,n)
                    ts.append(t)
                    grfs.append(grf)
                    dys.append(dy)

                    plt.subplot(211)
                    plt.plot(t,dy,'.',color=c,markersize=1)
                    plt.subplot(212)
                    plt.plot(t,grf,'.',color=c,markersize=1)

                tf = np.amin([np.amax(ts[i]) for i in range(len(ts))])
                t = np.linspace(0,tf,100)
                grfs = [np.interp(t,ts[i],grfs[i]) for i in range(len(grfs))]
                dys = [np.interp(t,ts[i],dys[i]) for i in range(len(dys))]

                grf = np.mean(grfs,axis=0)
                dy = np.mean(dys,axis=0)

                plt.subplot(211)
                plt.plot(t,dy,color=c)
                plt.subplot(212)
                plt.plot(t,grf,color=c)

                cs = jump.cs
                # cs['m'] = 0.028
                cs['k'] = k
                cs['a'] = a
                sol = jump.solve(cs)
                t_d = sol.t
                dy_d = sol.y[0,:]
                grf_d = jump.f_spring(sol.y[1,:],k,a,cs['ds'])
                plt.subplot(211)
                plt.plot(t_d,dy_d,'--',color=c)
                plt.subplot(212)
                plt.plot(t_d,grf_d,'--',color=c)
plt.show()
