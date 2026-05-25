import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
 
H = 20
W = 20
np.random.seed(7)
 
ii, jj = np.mgrid[0:H, 0:W].astype(float)
dist_main = np.sqrt((ii - 9.5) ** 2 + (jj - 9.5) ** 2)
signal = 3.6 * np.exp(-(dist_main ** 2) / 50)
noise = 0.34 * (2 * np.random.rand(H, W) - 1)
V = np.clip(signal + noise, 0, 4)
 
V_clip = np.clip(V, 0, 3)
V_norm = V_clip / 3
 
 
def quantize(v_norm, s):
    return np.clip(np.round(v_norm * s), 0, s) / s
 
 
def draw_continuous(ax, data, label, title, cmap, vmin, vmax, fmt, auto_text=False, fontsize=5):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12, pad=18, loc='center')
    ax.text(-0.12, 1.05, label, transform=ax.transAxes,
            fontsize=15, fontweight='bold', ha='left', va='bottom')
    ax.set_xlabel("W", labelpad=2)
    ax.set_ylabel("H", rotation=0, labelpad=10)
    cmap_obj = plt.colormaps[cmap]
    norm_obj = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mid = (vmin + vmax) / 2
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if auto_text:
                r, g, b, _ = cmap_obj(norm_obj(val))
                brightness = 0.299 * r + 0.587 * g + 0.114 * b
                color = "white" if brightness < 0.5 else "black"
            else:
                color = "white" if val > mid else "black"
            ax.text(j, i, fmt.format(val), ha="center", va="center",
                    fontsize=fontsize, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
 
 
def draw_spike(ax, data, s, label, title, fontsize=4):
    levels = np.linspace(0, 1, s + 1)
    if s == 4:
        colors = ['#3a4cc0', '#8daffd', '#dddcdb', '#f39879', '#b30326']
    else:
        base_cmap = plt.colormaps['coolwarm']
        sample_pts = np.linspace(0.0, 1.0, s + 1)
        colors = [base_cmap(v) for v in sample_pts]
    cmap = ListedColormap(colors)
 
    step = 1.0 / s
    boundaries = [-step / 2 + k * step for k in range(s + 2)]
    norm = BoundaryNorm(boundaries, cmap.N)
 
    im = ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=12, pad=18, loc='center')
    ax.text(-0.12, 1.05, label, transform=ax.transAxes,
            fontsize=15, fontweight='bold', ha='left', va='bottom')
    ax.set_xlabel("W", labelpad=2)
    ax.set_ylabel("H", rotation=0, labelpad=10)
 
    colors_rgba = [cmap(norm(v)) for v in levels]
    brightness = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b, _ in colors_rgba]
    val_to_bright = dict(zip(levels, brightness))
 
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            nearest = min(levels, key=lambda l: abs(l - val))
            txt_color = "black" if val_to_bright[nearest] > 0.55 else "white"
            if val >= 0.999:
                label = "1"
            elif val <= 0.001:
                label = "0"
            else:
                label = "{:.2f}".format(val)
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=fontsize, color=txt_color, fontweight="bold")
 
    if s <= 4:
        ticks = list(levels)
        tick_labels = ["{:.2f}".format(t) for t in levels]
    else:
        idx = np.linspace(0, s, 5).astype(int)
        ticks = [levels[k] for k in idx]
        tick_labels = ["{:.2f}".format(levels[k]) for k in idx]
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=ticks)
    cb.ax.set_yticklabels(tick_labels, fontsize=8)
 
 
fig = plt.figure(figsize=(13, 10.5), facecolor='white')
 
gs_top = fig.add_gridspec(1, 2, left=0.13, right=0.87,
                          top=0.93, bottom=0.56, wspace=0.30)
gs_bot = fig.add_gridspec(1, 2, left=0.13, right=0.87,
                          top=0.43, bottom=0.06, wspace=0.30)
 
ax1 = fig.add_subplot(gs_top[0, 0])
ax2 = fig.add_subplot(gs_top[0, 1])
ax3 = fig.add_subplot(gs_bot[0, 0])
ax4 = fig.add_subplot(gs_bot[0, 1])
 
for a in [ax1, ax2, ax3, ax4]:
    a.set_aspect('equal')
 
fig.text(0.5, 0.975, "Voxel Processing",
         fontsize=18, ha="center", weight="bold", color='#222222')
fig.text(0.5, 0.495, "Spike Quantization",
         fontsize=18, ha="center", weight="bold", color='#222222')
 
draw_continuous(ax1, V, "a", "Raw Positive Polarity Voxel V$^+$(h,w) at time step t",
                "coolwarm", 0, 4, "{:.1f}")
draw_continuous(ax2, V_norm, "b", "clip(V$^+$, 0, 3) + normalize",
                "YlGnBu", 0, 1, "{:.2f}")
 
V_s4 = quantize(V_norm, 4)
V_s8 = quantize(V_norm, 8)
 
draw_spike(ax3, V_s4, 4, "c", "clip + round + normalize  (S = 4)", fontsize=5)
draw_spike(ax4, V_s8, 8, "d", "clip + round + normalize  (S = 8)", fontsize=4)
 

plt.savefig("C:/Users/Administrator/Desktop/Nature Communications/edited/voxelspike/voxel_spike.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Done")