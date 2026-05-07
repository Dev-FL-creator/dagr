# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap, BoundaryNorm

# H = 20
# W = 20

# np.random.seed(7)

# ii, jj = np.mgrid[0:H, 0:W].astype(float)
# dist_main = np.sqrt((ii - 9.5) ** 2 + (jj - 9.5) ** 2)
# continuous_signal = 3.0 * np.exp(-(dist_main ** 2) / 50)

# S = 4
# quant_levels = np.clip(np.round(np.clip(continuous_signal, 0, 3) / 3 * S), 0, S)
# signal_snapped = quant_levels / S * 3

# noise = 0.34 * (2 * np.random.rand(H, W) - 1)
# V = signal_snapped + noise

# V_clip = np.clip(V, 0, 3)
# V_norm = V_clip / 3

# S = 4
# V_quant = np.clip(np.round(V_norm * S), 0, S)
# V_spike = V_quant / S

# quant_colors = ['#ffffff', '#b8b8b8', '#707070', '#383838', '#0a0a0a']
# quant_cmap = ListedColormap(quant_colors)
# quant_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], quant_cmap.N)

# spike_colors = ['#7b9ef8', '#c0d3f5', '#f1cab6', '#ed8467', '#b30326']
# spike_cmap = ListedColormap(spike_colors)
# spike_norm = BoundaryNorm([-0.01, 0.13, 0.38, 0.63, 0.88, 1.01], spike_cmap.N)


# def draw_continuous(ax, data, title, cmap, vmin, vmax, fmt, auto_text=False):
#     import matplotlib.cm as cm
#     import matplotlib.colors as mcolors
#     im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
#     ax.set_title(title, fontsize=12)
#     ax.set_xlabel("W")
#     ax.set_ylabel("H", rotation=0, labelpad=10)
#     mid = (vmin + vmax) / 2
#     cmap_obj = cm.get_cmap(cmap)
#     norm_obj = mcolors.Normalize(vmin=vmin, vmax=vmax)
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             val = data[i, j]
#             if auto_text:
#                 r, g, b, _ = cmap_obj(norm_obj(val))
#                 brightness = 0.299 * r + 0.587 * g + 0.114 * b
#                 color = "white" if brightness < 0.5 else "black"
#             else:
#                 color = "white" if val > mid else "black"
#             ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=5, color=color)
#     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# def draw_discrete(ax, data, title, cmap, norm, labels, fmt, dark_threshold=0.5):
#     im = ax.imshow(data, cmap=cmap, norm=norm)
#     ax.set_title(title, fontsize=12)
#     ax.set_xlabel("W")
#     ax.set_ylabel("H", rotation=0, labelpad=10)
#     colors_rgba = [cmap(norm(v)) for v in labels]
#     brightness = [0.299*r + 0.587*g + 0.114*b for r,g,b,_ in colors_rgba]
#     val_to_bright = dict(zip(labels, brightness))
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             val = data[i, j]
#             nearest = min(labels, key=lambda l: abs(l - val))
#             txt_color = "black" if val_to_bright[nearest] > dark_threshold else "white"
#             ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=5,
#                     color=txt_color, fontweight="bold")
#     cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=labels)
#     cb.ax.set_yticklabels([str(l) for l in labels])


# fig = plt.figure(figsize=(18, 11))
# gs = fig.add_gridspec(2, 6, top=0.88, bottom=0.07, hspace=0.65, wspace=0.6)

# ax1 = fig.add_subplot(gs[0, 0:2])
# ax2 = fig.add_subplot(gs[0, 2:4])
# ax3 = fig.add_subplot(gs[0, 4:6])
# ax4 = fig.add_subplot(gs[1, 1:3])
# ax5 = fig.add_subplot(gs[1, 3:5])

# fig.text(0.5, 0.95, "Voxel Processing", fontsize=18, ha="center", weight="bold")
# fig.text(0.5, 0.46, "Spike Quantization", fontsize=18, ha="center", weight="bold")

# draw_continuous(ax1, V, "(a) Raw voxel V(h,w) at time step t", "coolwarm", -1, 4, "{:.1f}")
# draw_continuous(ax2, V_clip, "(b) clip(V,0,3)", "YlGnBu", 0, 3, "{:.1f}")
# draw_continuous(ax3, V_norm, "(c) normalize /3", "viridis_r", 0, 1, "{:.2f}", auto_text=True)
# draw_discrete(ax4, V_quant, "(d) round + clip", quant_cmap, quant_norm, [0,1,2,3,4], "{:.0f}")

# bin_centers = [0.06, 0.255, 0.505, 0.755, 0.945]
# bin_labels  = ["0.0", "0.25", "0.5", "0.75", "1.0"]
# im5 = ax5.imshow(V_spike, cmap=spike_cmap, norm=spike_norm)
# ax5.set_title("(e) normalize /4", fontsize=12)
# ax5.set_xlabel("W")
# ax5.set_ylabel("H", rotation=0, labelpad=10)
# colors_rgba5 = [spike_cmap(spike_norm(v)) for v in [0.0, 0.25, 0.5, 0.75, 1.0]]
# brightness5  = [0.299*r + 0.587*g + 0.114*b for r,g,b,_ in colors_rgba5]
# val_to_bright5 = dict(zip([0.0, 0.25, 0.5, 0.75, 1.0], brightness5))
# for i in range(V_spike.shape[0]):
#     for j in range(V_spike.shape[1]):
#         val = V_spike[i, j]
#         nearest = min([0.0, 0.25, 0.5, 0.75, 1.0], key=lambda l: abs(l - val))
#         txt_color = "black" if val_to_bright5[nearest] > 0.5 else "white"
#         ax5.text(j, i, "1" if val >= 0.999 else "{:.2f}".format(val), ha="center", va="center", fontsize=4,
#                  color=txt_color, fontweight="bold")
# cb5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, ticks=bin_centers)
# cb5.ax.set_yticklabels(bin_labels)

# plt.savefig("voxel_spike.png", dpi=150, bbox_inches="tight")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
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
 
S = 4
V_quant = np.clip(np.round(V_norm * S), 0, S)
V_spike = V_quant / S
 
quant_colors = ['#ffffff', '#b8b8b8', '#707070', '#383838', '#0a0a0a']
quant_cmap = ListedColormap(quant_colors)
quant_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], quant_cmap.N)
 
spike_colors = ['#3a4cc0', '#8daffd', '#dddcdb', '#f39879', '#b30326']
spike_cmap = ListedColormap(spike_colors)
spike_norm = BoundaryNorm([-0.01, 0.13, 0.38, 0.63, 0.88, 1.01], spike_cmap.N)
 
 
def draw_continuous(ax, data, title, cmap, vmin, vmax, fmt, auto_text=False):
    import matplotlib.colors as mcolors
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12, pad=6)
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
            ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=5, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
 
 
def draw_discrete(ax, data, title, cmap, norm, labels, fmt, dark_threshold=0.5):
    im = ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=12, pad=6)
    ax.set_xlabel("W", labelpad=2)
    ax.set_ylabel("H", rotation=0, labelpad=10)
    colors_rgba = [cmap(norm(v)) for v in labels]
    brightness = [0.299*r + 0.587*g + 0.114*b for r, g, b, _ in colors_rgba]
    val_to_bright = dict(zip(labels, brightness))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            nearest = min(labels, key=lambda l: abs(l - val))
            txt_color = "black" if val_to_bright[nearest] > dark_threshold else "white"
            ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=5,
                    color=txt_color, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=labels)
    cb.ax.set_yticklabels([str(l) for l in labels])
 
 
fig = plt.figure(figsize=(13, 10), facecolor='white')
 
gs_top = fig.add_gridspec(1, 3, left=0.06, right=0.98,
                          top=0.94, bottom=0.60, wspace=0.30)
gs_bot = fig.add_gridspec(1, 2, left=0.215, right=0.785,
                          top=0.47, bottom=0.09, wspace=0.30)
 
ax1 = fig.add_subplot(gs_top[0, 0])
ax2 = fig.add_subplot(gs_top[0, 1])
ax3 = fig.add_subplot(gs_top[0, 2])
ax4 = fig.add_subplot(gs_bot[0, 0])
ax5 = fig.add_subplot(gs_bot[0, 1])
 
fig.text(0.5, 0.975, "Voxel Processing",
         fontsize=18, ha="center", weight="bold", color='#222222')
fig.text(0.5, 0.485, "Spike Quantization",
         fontsize=18, ha="center", weight="bold", color='#222222')
 
draw_continuous(ax1, V, "(a) Raw Positive Polarity Voxel V$^+$(h,w) at time step t", "coolwarm", 0, 4, "{:.1f}")
draw_continuous(ax2, V_clip, "(b) clip(V$^+$,0,3)", "YlGnBu", 0, 3, "{:.1f}")
draw_continuous(ax3, V_norm, "(c) normalize /3", "viridis_r", 0, 1, "{:.2f}", auto_text=True)
draw_discrete(ax4, V_quant, "(d) clip + round", quant_cmap, quant_norm, [0,1,2,3,4], "{:.0f}")
 
bin_centers = [0.06, 0.255, 0.505, 0.755, 0.945]
bin_labels = ["0.0", "0.25", "0.5", "0.75", "1.0"]
im5 = ax5.imshow(V_spike, cmap=spike_cmap, norm=spike_norm)
ax5.set_title("(e) normalize /4", fontsize=12, pad=6)
ax5.set_xlabel("W", labelpad=2)
ax5.set_ylabel("H", rotation=0, labelpad=10)
colors_rgba5 = [spike_cmap(spike_norm(v)) for v in [0.0, 0.25, 0.5, 0.75, 1.0]]
brightness5 = [0.299*r + 0.587*g + 0.114*b for r, g, b, _ in colors_rgba5]
val_to_bright5 = dict(zip([0.0, 0.25, 0.5, 0.75, 1.0], brightness5))
for i in range(V_spike.shape[0]):
    for j in range(V_spike.shape[1]):
        val = V_spike[i, j]
        nearest = min([0.0, 0.25, 0.5, 0.75, 1.0], key=lambda l: abs(l - val))
        txt_color = "black" if val_to_bright5[nearest] > 0.5 else "white"
        ax5.text(j, i, "1" if val >= 0.999 else "{:.2f}".format(val),
                 ha="center", va="center", fontsize=4, color=txt_color, fontweight="bold")
cb5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.03, ticks=bin_centers)
cb5.ax.set_yticklabels(bin_labels)

plt.savefig("C:/Users/Administrator/Desktop/Nature Communications/edited/voxelspike/voxel_spike.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Done")