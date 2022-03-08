import matplotlib.pyplot as plt

def set_default():
    fs = 8
    lw = 0.6
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = fs
    plt.rcParams['axes.titlesize'] = fs
    plt.rcParams['axes.linewidth'] = lw
    plt.rcParams['xtick.labelsize'] = fs
    plt.rcParams['ytick.labelsize'] = fs
    plt.rcParams['xtick.major.width'] = lw
    plt.rcParams['ytick.major.width'] = lw
    plt.rcParams['patch.linewidth'] = lw
