import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

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

pad = 0.005
def savefig(name,fig,pdf=None):
    fig_size = fig.get_size_inches()
    bbox = mtransforms.Bbox([[0-pad,0-pad],[fig_size[0]+pad,fig_size[1]+pad]])

    if pdf is None:
        fig.savefig(name,bbox_inches=bbox)
    else:
        pdf.savefig(bbox_inches=bbox)
