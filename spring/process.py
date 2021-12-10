import numpy as np
import matplotlib.pyplot as plt
import data

DEG_PER_COUNT = 0.088
ZERO_FORCE_TH = 1e-3
POS_MID = 2068

PI = np.pi
DEG_2_RAD = PI/180

def remove_gap(ps0,tzs0,gap):
    gap_count = int(gap/DEG_PER_COUNT)
    ps = ps0 + (tzs0>0)*gap_count/2 - (tzs0<0)*gap_count/2 # accounts gap

    non_zero_idx = np.logical_or(tzs0 > ZERO_FORCE_TH, tzs0 < -ZERO_FORCE_TH) # remove zero force
    ps = ps[non_zero_idx]
    tzs = tzs0[non_zero_idx]

    # small_angle_idx = np.abs(ps-POS_MID) < 5/DEG_PER_COUNT
    # ps = ps[small_angle_idx]
    # tzs = tzs[small_angle_idx]

    return ps, tzs

def base2virtual(ps,tzs,R):
    thetas = []
    ls = []
    alphas = []
    Fs = []
    Fpls = []
    Mls = []
    alphaps = []
    for p,tz in zip(ps,tzs):
        theta = (p-POS_MID)*DEG_PER_COUNT*DEG_2_RAD
        l = R/np.cos(theta)/2

        cos_alpha = (l**2+l**2-R**2)/(2*l*l)
        sin_alpha = R/l*np.sin(theta)

        alpha = np.arctan2(sin_alpha,cos_alpha)

        F = -tz/R
        Fpl = F*np.sin(alpha-PI/2+theta)
        Ml = Fpl*l # Moment required to deform to this angle

        alphap = np.sign(alpha)*PI-alpha

        thetas.append(theta/DEG_2_RAD)
        ls.append(l)
        alphas.append(alpha/DEG_2_RAD)
        Fs.append(F)
        Fpls.append(Fpl)
        Mls.append(Ml)
        alphaps.append(alphap)

    # plt.figure()
    # plt.plot(alphaps,Mls)
    # plt.show()

    return np.array(alphaps),np.array(Mls)

def k(t,w,l):
    gap = np.arctan2(0.9,l)/np.pi*180*2

    ps0 = []
    tzs0 = []
    for sample in [1]:
        fs = data.read('../data/{:d}mil_{:d}mm_{:d}mm_{:d}.csv'.format(t,l+15,w,sample))
        ps0.append(np.array(fs['p']))
        tzs0.append(np.array(fs['tz']))

    ps0 = np.concatenate(ps0)
    tzs0 = np.concatenate(tzs0)
    ps, tzs = remove_gap(ps0,tzs0,gap)

    # # Raw and adjusted data
    # plt.figure()
    # plt.plot(ps0,tzs0,label='raw')
    # plt.plot(ps,tzs,'.',label='adjusted')
    # plt.xlabel('Position Count (0.088deg/count)')
    # plt.ylabel('Base Torque [Nm]')
    # plt.legend()
    # plt.show()

    pv, tv = base2virtual(ps,tzs,l/1000)
    coeff = np.polyfit(pv,tv,1)
    k,b = coeff
    pvfit = np.array([np.amin(pv),np.amax(pv)])
    tvfit = np.polyval(coeff,pvfit)

    # k = np.linalg.lstsq(pv.reshape(-1,1),tv,rcond=None)[0][0]
    # pvfit = np.array([np.amin(pv),np.amax(pv)])
    # tvfit = pvfit*k

    return k,b,(pv,tv,pvfit,tvfit)
