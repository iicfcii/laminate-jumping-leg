import numpy as np
import matplotlib.pyplot as plt
import data

DEG_PER_COUNT = 0.088
GAP_COUNT = int(4/DEG_PER_COUNT)
ZERO_FORCE_TH = 1e-3
POS_MID = 2068
R = 25/1000

PI = np.pi
DEG_2_RAD = PI/180

def remove_gap(ps0,tzs0):
    ps = ps0 + (tzs0>0)*GAP_COUNT/2 - (tzs0<0)*GAP_COUNT/2 # accounts gap
    non_zero_idx = np.logical_or(tzs0 > ZERO_FORCE_TH, tzs0 < -ZERO_FORCE_TH) # remove zero force
    ps = ps[non_zero_idx]
    tzs = tzs0[non_zero_idx]

    return ps, tzs

def base2virtual(ps,tzs):
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
