import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

np.random.seed(42)

NQ, NK, C = 8, 8, 16
D = 8
scale = D ** -0.5

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def spike(x):
    return np.clip(x, 0.0, 1.0)

signal_q = np.zeros((NQ, C))
for i in range(4):
    signal_q[i*4:(i+1)*4, i*4:(i+1)*4] = 1.2
signal_k = signal_q.copy()

raw_q = signal_q + 0.5 * np.random.randn(NQ, C)
raw_k = signal_k + 0.5 * np.random.randn(NK, C)
raw_v = signal_q + 0.1 * np.random.randn(NQ, C)

logits_raw = raw_q[:, :D] @ raw_k[:, :D].T * scale
attn_raw   = softmax(logits_raw)

q_spk = spike(raw_q)
k_spk = spike(raw_k)
v_spk = spike(raw_v)
logits_spk = q_spk[:, :D] @ k_spk[:, :D].T * scale
attn_spk   = softmax(logits_spk)

entropy_raw  = -(attn_raw * np.log(attn_raw + 1e-9)).sum(axis=-1).mean()
entropy_spk  = -(attn_spk * np.log(attn_spk + 1e-9)).sum(axis=-1).mean()
collapse_raw = (attn_raw.max(axis=-1) > 0.5).mean() * 100
collapse_spk = (attn_spk.max(axis=-1) > 0.5).mean() * 100


def draw(ax, data, title, cmap, vmin, vmax, fontsize=9, annotation=None):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(title, fontsize=11.5, pad=10, weight='medium')
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.tick_params(axis='both', which='major', labelsize=8.5, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, "{:.2f}".format(val), ha="center", va="center",
                    fontsize=fontsize, color="black")
    if annotation:
        ax.text(0.99, -0.10, annotation, transform=ax.transAxes,
                fontsize=8.5, color='#333333', ha='right', va='top',
                style='italic')
    return im


noise_levels = np.linspace(0.0, 2.0, 40)
N_RUNS = 50
collapse_raw_runs = np.zeros((N_RUNS, len(noise_levels)))
collapse_spk_runs = np.zeros((N_RUNS, len(noise_levels)))
entropy_raw_runs  = np.zeros((N_RUNS, len(noise_levels)))
entropy_spk_runs  = np.zeros((N_RUNS, len(noise_levels)))
logvar_raw_runs   = np.zeros((N_RUNS, len(noise_levels)))
logvar_spk_runs   = np.zeros((N_RUNS, len(noise_levels)))

for run in range(N_RUNS):
    for ni, nl in enumerate(noise_levels):
        nq = signal_q + nl * np.random.randn(NQ, C)
        nk = signal_k + nl * np.random.randn(NK, C)
        l_r = nq[:, :D] @ nk[:, :D].T * scale
        a_r = softmax(l_r)
        collapse_raw_runs[run, ni] = (a_r.max(axis=-1) > 0.5).mean() * 100
        entropy_raw_runs[run, ni]  = -(a_r * np.log(a_r + 1e-9)).sum(axis=-1).mean()
        logvar_raw_runs[run, ni]   = np.var(l_r)
        sq = spike(nq); sk = spike(nk)
        l_s = sq[:, :D] @ sk[:, :D].T * scale
        a_s = softmax(l_s)
        collapse_spk_runs[run, ni] = (a_s.max(axis=-1) > 0.5).mean() * 100
        entropy_spk_runs[run, ni]  = -(a_s * np.log(a_s + 1e-9)).sum(axis=-1).mean()
        logvar_spk_runs[run, ni]   = np.var(l_s)

collapse_mean_raw = collapse_raw_runs.mean(0); collapse_std_raw = collapse_raw_runs.std(0)
collapse_mean_spk = collapse_spk_runs.mean(0); collapse_std_spk = collapse_spk_runs.std(0)
entropy_mean_raw  = entropy_raw_runs.mean(0);  entropy_std_raw  = entropy_raw_runs.std(0)
entropy_mean_spk  = entropy_spk_runs.mean(0);  entropy_std_spk  = entropy_spk_runs.std(0)
logvar_mean_raw   = logvar_raw_runs.mean(0);   logvar_std_raw   = logvar_raw_runs.std(0)
logvar_mean_spk   = logvar_spk_runs.mean(0);   logvar_std_spk   = logvar_spk_runs.std(0)


plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titleweight': 'medium',
    'axes.edgecolor': '#444444',
    'axes.linewidth': 0.8,
    'xtick.color': '#444444',
    'ytick.color': '#444444',
})

fig = plt.figure(figsize=(16, 24), facecolor='white')

gs = gridspec.GridSpec(
    6, 2, figure=fig,
    left=0.05, right=0.97, top=0.96, bottom=0.04,
    hspace=0.32, wspace=0.22,
    height_ratios=[1.0, 1.0, 1.0, 1.0, 0.85, 0.85]
)

fig.text(0.27, 0.978, "Standard Cross-Attention",
         fontsize=15, ha="center", weight="bold", color="#b30326")
fig.text(0.74, 0.978, "Spike-Driven Cross-Attention",
         fontsize=15, ha="center", weight="bold", color="#1a1a6e")



ax_q_raw = fig.add_subplot(gs[0, 0])
ax_q_spk = fig.add_subplot(gs[0, 1])
ax_k_raw = fig.add_subplot(gs[1, 0])
ax_k_spk = fig.add_subplot(gs[1, 1])
ax_l_raw = fig.add_subplot(gs[2, 0])
ax_l_spk = fig.add_subplot(gs[2, 1])
ax_a_raw = fig.add_subplot(gs[3, 0])
ax_a_spk = fig.add_subplot(gs[3, 1])
ax_col   = fig.add_subplot(gs[4, :])
ax_ent   = fig.add_subplot(gs[5, 0])
ax_lvar  = fig.add_subplot(gs[5, 1])

qk_v = max(abs(raw_q[:, :D]).max(), abs(raw_k[:, :D]).max(),
           q_spk[:, :D].max(), k_spk[:, :D].max())

cbar_kw = dict(fraction=0.04, pad=0.025, aspect=18)

im = draw(ax_q_raw, raw_q[:, :D], "(a) Q activations  [continuous + noise]",
          "RdBu_r", -qk_v, qk_v,
          annotation=f"range [{raw_q[:,:D].min():.2f}, {raw_q[:,:D].max():.2f}]  noisy")
plt.colorbar(im, ax=ax_q_raw, **cbar_kw)

im = draw(ax_q_spk, q_spk[:, :D], "(b) Q activations  [spike clamp to (0,1)]",
          "RdBu_r", -qk_v, qk_v,
          annotation="range [0.0, 1.0]  negative noise removed")
plt.colorbar(im, ax=ax_q_spk, **cbar_kw)

im = draw(ax_k_raw, raw_k[:, :D], "(c) K activations  [continuous + noise]",
          "RdBu_r", -qk_v, qk_v,
          annotation=f"range [{raw_k[:,:D].min():.2f}, {raw_k[:,:D].max():.2f}]  noisy")
plt.colorbar(im, ax=ax_k_raw, **cbar_kw)

im = draw(ax_k_spk, k_spk[:, :D], "(d) K activations  [spike clamp to (0,1)]",
          "RdBu_r", -qk_v, qk_v,
          annotation="range [0.0, 1.0]  negative noise removed")
plt.colorbar(im, ax=ax_k_spk, **cbar_kw)

l_v = max(abs(logits_raw).max(), abs(logits_spk).max())
im = draw(ax_l_raw, logits_raw, "(e) Attention logits  [continuous]",
          "coolwarm", -l_v, l_v,
          annotation=f"std = {logits_raw.std():.2f}  high variance \u2192 collapse risk")
plt.colorbar(im, ax=ax_l_raw, **cbar_kw)

im = draw(ax_l_spk, logits_spk, "(f) Attention logits  [bounded]",
          "coolwarm", -l_v, l_v,
          annotation=f"std = {logits_spk.std():.2f}  bounded logits \u2192 stable attention")
plt.colorbar(im, ax=ax_l_spk, **cbar_kw)

a_vmax = max(attn_raw.max(), attn_spk.max())
im = draw(ax_a_raw, attn_raw, "(g) Attention weights  [collapse risk]",
          "YlOrRd", 0, a_vmax,
          annotation=f"entropy={entropy_raw:.2f}  collapse={collapse_raw:.0f}%")
plt.colorbar(im, ax=ax_a_raw, **cbar_kw)

im = draw(ax_a_spk, attn_spk, "(h) Attention weights  [stable]",
          "YlOrRd", 0, a_vmax,
          annotation=f"entropy={entropy_spk:.2f}  collapse={collapse_spk:.0f}%")
plt.colorbar(im, ax=ax_a_spk, **cbar_kw)


def plot_curve(ax, x, m_r, s_r, m_s, s_s, ylabel, title_label, legend_loc="upper left"):
    ax.plot(x, m_r, color="#b30326", linewidth=2.2, zorder=3)
    ax.fill_between(x, m_r - s_r, m_r + s_r, alpha=0.22, color="#b30326",
                    linewidth=0, zorder=2)
    ax.plot(x, m_s, color="#1a1a6e", linewidth=2.2, zorder=3)
    ax.fill_between(x, m_s - s_s, m_s + s_s, alpha=0.22, color="#1a1a6e",
                    linewidth=0, zorder=2)
    ax.set_xlabel("Input noise level (std)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title_label + "\n(mean \u00b1 1 std over 50 runs)",
                 fontsize=11, pad=8, weight='medium')
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax.grid(axis='x', alpha=0.15, linewidth=0.6)
    ax.set_xlim(0, 2.0)
    ax.tick_params(axis='both', which='major', labelsize=9)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    legend_elements = [
        Line2D([0], [0], color="#b30326", linewidth=2.2, label="Standard (mean)"),
        Patch(facecolor="#b30326", alpha=0.32, label="Standard \u00b1 1 std"),
        Line2D([0], [0], color="#1a1a6e", linewidth=2.2, label="Spike (mean)"),
        Patch(facecolor="#1a1a6e", alpha=0.32, label="Spike \u00b1 1 std"),
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, loc=legend_loc,
              frameon=True, framealpha=0.9, edgecolor='#cccccc')


plot_curve(ax_col, noise_levels,
           collapse_mean_raw, collapse_std_raw,
           collapse_mean_spk, collapse_std_spk,
           "Attention collapse rate (%)",
           "(i) Attention collapse rate vs. noise level")
ax_col.set_ylim(-2, 105)
ax_col.axhline(0, color='gray', linewidth=0.6, linestyle='--', alpha=0.6)

plot_curve(ax_ent, noise_levels,
           entropy_mean_raw, entropy_std_raw,
           entropy_mean_spk, entropy_std_spk,
           "Attention entropy (higher = less collapse)",
           "(j) Attention entropy vs. noise level",
           legend_loc="lower left")

plot_curve(ax_lvar, noise_levels,
           logvar_mean_raw, logvar_std_raw,
           logvar_mean_spk, logvar_std_spk,
           "Logit variance (lower = stable)",
           "(k) Attention logit variance vs. noise level")

plt.savefig("C:/Users/Administrator/Desktop/Nature Communications/edited/spike cross attention/attn_compare.png",
            dpi=150, bbox_inches="tight", facecolor='white')
plt.close(fig)
print("Done")