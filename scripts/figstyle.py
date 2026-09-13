"""Shared figure style. IEEE two-column: 3.5in single, 7.16in double."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL1, COL2 = 3.5, 7.16

# Deliberately not the matplotlib default cycle: two saturated hues that stay
# distinguishable in greyscale (lightness 45 vs 68) for print reviewers.
OURS   = '#1b4f72'   # deep blue   -- the method
CTRL   = '#c0532b'   # burnt orange-- the ablation control
GT     = '#2d7d46'   # green       -- ground truth
MUTED  = '#8a8f98'

def setup():
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': 0.6, 'grid.linewidth': 0.4, 'lines.linewidth': 1.3,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'axes.grid': True, 'grid.alpha': 0.25,
    })
