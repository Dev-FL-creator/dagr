import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

np.random.seed(42)

S = 4.0

H, W = 64, 64

voxel_raw = np.random.randn(H, W) * 2.5
voxel_raw += np.sin(np.linspace(0, 3 * np.pi, H))[:, None]
voxel_raw += np.cos(np.linspace(0, 2 * np.pi, W))[None, :]

x = voxel_raw.astype(np.float32)
x_clipped = np.clip(x, 0.0, S)
x_rounded = np.round(x_clipped)
x_normalized = x_rounded / S

steps = {
    "Step 1 — Raw Voxel\n(continuous activation $x$)": x,
    "Step 2 — After Clip\n$\\mathrm{Clip}(x,\\ 0,\\ S)$": x_clipped,
    "Step 3 — After Round\n$\\mathrm{round}(\\cdot)$": x_rounded,
    "Step 4 — Normalized Output\n$s = \\frac{1}{S}\\,\\mathrm{Clip}(\\mathrm{round}(x),\\ 0,\\ S)$": x_normalized,
}

fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor("#0e0e0e")

outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.55)
top_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.35)
bot_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.38)

fig.text(
    0.5, 0.97,
    "MultiSpike / SFA Quantization — Step-by-Step Visualization",
    ha="center", va="top",
    fontsize=17, fontweight="bold",
    color="white", fontfamily="monospace"
)

CMAPS   = ["RdBu_r", "YlOrRd", "Blues", "Greens"]
BORDERS = ["#888888", "#f0a500", "#5bc8f5", "#66e09a"]

data_list = list(steps.items())

for col, (title, data) in enumerate(data_list):
    ax = fig.add_subplot(top_gs[col])
    ax.set_facecolor("#1a1a1a")

    vmax = max(abs(data.min()), abs(data.max())) if col == 0 else data.max()
    vmin = -vmax if col == 0 else 0.0

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if col == 0 else None
    im = ax.imshow(data, cmap=CMAPS[col], norm=norm, interpolation="nearest", aspect="auto")

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDERS[col])
        spine.set_linewidth(2.2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=8)
    cbar.outline.set_edgecolor(BORDERS[col])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(title, color="white", fontsize=10.5, pad=10, fontfamily="monospace")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(True)

    stats = f"min={data.min():.2f}  max={data.max():.2f}  mean={data.mean():.2f}"
    ax.set_xlabel(stats, color="#aaaaaa", fontsize=7.5, fontfamily="monospace")

    if col > 0:
        unique_vals = np.unique(np.round(data, 4))
        info = f"Unique levels: {len(unique_vals)}"
        ax.set_ylabel(info, color=BORDERS[col], fontsize=8, fontfamily="monospace")


ax_hist = fig.add_subplot(bot_gs[0])
ax_hist.set_facecolor("#1a1a1a")
for spine in ax_hist.spines.values():
    spine.set_edgecolor("#888888")
    spine.set_linewidth(1.5)
colors_hist = ["#888888", "#f0a500", "#5bc8f5", "#66e09a"]
labels_hist = ["Raw $x$", "Clipped", "Rounded", "Normalized $s$"]
for (title, data), col_h, lbl in zip(data_list, colors_hist, labels_hist):
    ax_hist.hist(data.ravel(), bins=60, alpha=0.55, color=col_h, label=lbl, density=True)
ax_hist.set_title("Value Distribution — All Steps", color="white", fontsize=11, fontfamily="monospace", pad=8)
ax_hist.set_xlabel("Value", color="#aaaaaa", fontsize=9)
ax_hist.set_ylabel("Density", color="#aaaaaa", fontsize=9)
ax_hist.tick_params(colors="white", labelsize=8)
ax_hist.legend(fontsize=8, facecolor="#222222", labelcolor="white", edgecolor="#555555")
ax_hist.set_facecolor("#1a1a1a")
ax_hist.grid(axis="y", color="#333333", linewidth=0.6)


ax_diff1 = fig.add_subplot(bot_gs[1])
ax_diff1.set_facecolor("#1a1a1a")
diff_clip = x_clipped - x
abs_max = np.abs(diff_clip).max()
im2 = ax_diff1.imshow(diff_clip, cmap="PiYG", vmin=-abs_max, vmax=abs_max, interpolation="nearest", aspect="auto")
for spine in ax_diff1.spines.values():
    spine.set_edgecolor("#cc66ff")
    spine.set_linewidth(2.0)
cbar2 = fig.colorbar(im2, ax=ax_diff1, fraction=0.045, pad=0.03)
cbar2.ax.yaxis.set_tick_params(color="white", labelsize=8)
cbar2.outline.set_edgecolor("#cc66ff")
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="white")
ax_diff1.set_title("Δ Clip effect\n(Clipped − Raw)", color="white", fontsize=10.5, fontfamily="monospace", pad=8)
ax_diff1.tick_params(colors="white", labelsize=7)
neg_pct = (diff_clip < 0).mean() * 100
ax_diff1.set_xlabel(f"Zeroed (neg): {neg_pct:.1f}% of pixels", color="#cc66ff", fontsize=8, fontfamily="monospace")


ax_spike = fig.add_subplot(bot_gs[2])
ax_spike.set_facecolor("#1a1a1a")
for spine in ax_spike.spines.values():
    spine.set_edgecolor("#66e09a")
    spine.set_linewidth(2.0)

levels = np.array([0, 1/S, 2/S, 3/S, 1.0])
level_names = [f"0/{int(S)}", f"1/{int(S)}", f"2/{int(S)}", f"3/{int(S)}", f"{int(S)}/{int(S)}"]
counts = [(x_normalized.ravel() == lv).sum() for lv in levels]
bars = ax_spike.bar(level_names, counts, color=["#444", "#5bc8f5", "#f0a500", "#f07050", "#66e09a"],
                    edgecolor="#333333", linewidth=1.2)
for bar, cnt in zip(bars, counts):
    ax_spike.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                  str(cnt), ha="center", va="bottom", color="white", fontsize=9, fontfamily="monospace")

ax_spike.set_title(f"Spike Level Distribution\n$s \\in \\{{0, 1/S, 2/S, 3/S, 1\\}}$   S={int(S)}", 
                   color="white", fontsize=10.5, fontfamily="monospace", pad=8)
ax_spike.set_xlabel("Spike Level", color="#aaaaaa", fontsize=9)
ax_spike.set_ylabel("Pixel Count", color="#aaaaaa", fontsize=9)
ax_spike.tick_params(colors="white", labelsize=9)
ax_spike.grid(axis="y", color="#333333", linewidth=0.6)
ax_spike.set_facecolor("#1a1a1a")

formula = (
    r"Forward:  $s(t) = \frac{1}{S}\,\mathrm{Clip}(\mathrm{round}(x(t)),\;0,\;S)$"
    "     |     "
    r"STE Backward:  $\frac{\partial s}{\partial x} = \mathbf{1}_{[0 \leq x \leq S]}$"
    f"     |     S = {int(S)}"
)
fig.text(0.5, 0.005, formula, ha="center", va="bottom",
         fontsize=11, color="#cccccc", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a1a", edgecolor="#444444", linewidth=1.5))

import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multispike_visualization.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved to {out_path}")