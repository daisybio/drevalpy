"""
Create a UMAP embedding plot highlighting specific drugs from the CTRPv2 dataset. (Currently set to highlight none.).

First, run make_umap_embedding.py to generate the embedding data file.npz, then run this script to create the plot.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

highlight_drugs = []  # ["LRRK2-IN-1", "Vorinostat", "avicin D", "birinapant"]
pearson_values = {
    "birinapant": 0.023,
    "avicin D": 0.095,
    "LRRK2-IN-1": 0.660,
    "Vorinostat": 0.647,
}
k = 0  # number of nearest neighbors

COLOR_BG = "#3D98D3"
COLOR_TRAIN = "#EC0B88"
COLOR_TEST = "#000000"
font_adder = 8
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 15 + font_adder,
        "axes.labelsize": 15 + font_adder,
        "xtick.labelsize": 15 + font_adder,
        "ytick.labelsize": 15 + font_adder,
        "legend.fontsize": 15 + font_adder,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
    }
)

data = np.load("LDO_embedding_data.npz", allow_pickle=True)
X_focus_2d = data["X_focus_2d"]
X_bg_2d = data["X_bg_2d"]
focus_is_train = data["focus_is_train"]
names = data["focus_drug_names"]

train_mask = focus_is_train.astype(bool)
test_mask = ~train_mask

mpl.rcParams["agg.path.chunksize"] = 10000
fig, ax = plt.subplots(figsize=(15, 15), dpi=400)

ax.scatter(
    X_bg_2d[:, 0],
    X_bg_2d[:, 1],
    s=2,
    alpha=0.9,
    color=COLOR_BG,
    linewidths=0,
    edgecolors="none",
    zorder=0,
    rasterized=True,
    label="Random ZINC Molecules",
)
ax.scatter(
    X_focus_2d[train_mask, 0],
    X_focus_2d[train_mask, 1],
    s=8,
    alpha=0.95,
    color=COLOR_TRAIN,
    zorder=1,
    label="CTRPv2 Train Drugs",
)
ax.scatter(
    X_focus_2d[test_mask, 0],
    X_focus_2d[test_mask, 1],
    s=21,
    alpha=1,
    color=COLOR_TEST,
    zorder=1,
    marker="^",
    label="CTRPv2 Test Drugs",
)

for drug_name in highlight_drugs:
    if drug_name not in names:
        print(f"[warn] {drug_name!r} not found, skipping")
        continue

    drug_idx = np.where(names == drug_name)[0][0]
    is_train = bool(focus_is_train[drug_idx])

    same_mask = train_mask if is_train else test_mask
    same_idx = np.where(same_mask)[0]
    dists = np.linalg.norm(X_focus_2d[same_idx] - X_focus_2d[drug_idx], axis=1)
    closest_idx = same_idx[np.argsort(dists)[:k]]

    pearson_r = pearson_values.get(drug_name, None)
    label_text = f"{drug_name}  (Pearson={pearson_r:.3f})" if pearson_r is not None else drug_name

    ax.scatter(
        X_focus_2d[drug_idx, 0],
        X_focus_2d[drug_idx, 1],
        s=250,
        edgecolors="black",
        facecolors="none",
        linewidths=2.5,
        zorder=5,
    )
    ax.text(
        X_focus_2d[drug_idx, 0] + 0.5,
        X_focus_2d[drug_idx, 1] + 0.5,
        label_text,
        color="black",
        fontsize=18,
        weight="bold",
        zorder=6,
    )

    for i, idx in enumerate(closest_idx):
        ax.plot(
            [X_focus_2d[drug_idx, 0], X_focus_2d[idx, 0]],
            [X_focus_2d[drug_idx, 1], X_focus_2d[idx, 1]],
            color="gray",
            linewidth=1.0,
            alpha=0.6,
            zorder=2,
        )
        offset_x = (i - 1) * 0.3
        offset_y = (1 - abs(i - 1)) * 0.3
        ax.text(
            X_focus_2d[idx, 0] + offset_x,
            X_focus_2d[idx, 1] + offset_y,
            names[idx],
            color="gray",
            fontsize=15,
            zorder=6,
        )

ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.legend(scatterpoints=1, markerscale=4)
fig.tight_layout()
out_path = "LDO_embedding_umap_with_list_labels_r2.pdf"
fig.savefig(out_path, dpi=400, bbox_inches="tight")
plt.close(fig)
print(f"saved {out_path}")
