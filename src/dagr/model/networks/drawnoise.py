import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# =========================
# 1. 像素风 Ring
# =========================
H, W = 14, 14

yy, xx = np.mgrid[0:H, 0:W]

cx, cy = 6.5, 6.5

dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)

# 圆环
clean = ((dist > 3.0) & (dist < 5.0)).astype(float)

# =========================
# 2. noisy版本
# =========================
np.random.seed(7)

noisy = clean.copy()

background_mask = (clean == 0)

# 背景浅紫噪声
noise_levels = [0.10, 0.22, 0.35]
noise_probs  = [0.65, 0.25, 0.10]

noise = np.random.choice(
    noise_levels,
    size=clean.shape,
    p=noise_probs
)

noisy[background_mask] += noise[background_mask]

# =========================
# 3. 主体周围增加更深噪声
# =========================
for i in range(clean.shape[0]):
    for j in range(clean.shape[1]):

        if clean[i, j] == 1:

            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:

                    ni, nj = i + di, j + dj

                    if (
                        0 <= ni < clean.shape[0]
                        and 0 <= nj < clean.shape[1]
                        and clean[ni, nj] == 0
                    ):

                        r = np.random.rand()

                        # 深紫噪声（更明显）
                        if r < 0.12:
                            noisy[ni, nj] = 0.78

                        # 中等紫色噪声
                        elif r < 0.28:
                            noisy[ni, nj] = 0.55

# 防止超过范围
noisy = np.clip(noisy, 0, 1)

# =========================
# 4. 放大像素
# =========================
scale = 22

clean_big = np.kron(clean, np.ones((scale, scale)))
noisy_big = np.kron(noisy, np.ones((scale, scale)))

# =========================
# 5. 紫色 colormap
# =========================
colors = [
    "#f3effd",   # 很淡背景
    "#ddd0f8",
    "#b89aec",
    "#7b5fc7",
    "#43206f"    # 深紫主体
]

cmap = ListedColormap(colors)

# =========================
# 6. 绘图
# =========================
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

axes[0].imshow(noisy_big, cmap=cmap, vmin=0, vmax=1)
axes[1].imshow(clean_big, cmap=cmap, vmin=0, vmax=1)

axes[0].set_title("Noisy Input", fontsize=13)
axes[1].set_title("Clean Output", fontsize=13)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

plt.subplots_adjust(wspace=0.12)

plt.show()