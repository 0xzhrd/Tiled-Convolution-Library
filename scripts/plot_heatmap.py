#!/usr/bin/env python3
"""
Turn bench CSVs into the project's figures and the crossover table.

  python3 scripts/plot_heatmap.py results/4060.csv [results/orin.csv ...] --out docs/assets

For each CSV (one device) and each (border mode, batch size) it writes:
  winner_<tag>_<border>_b<batch>.png   fastest runner per (filter size, input size)
  ratio_<tag>_<border>_b<batch>.png    halo vs cache: how many times faster, and which way
and one crossover_<tag>.csv: per (border, batch, input size), the first radius at which the
halo variant beats the cache variant, which is the number SelectorConfig::cache_max_radius
should be set from.

Colour: three categories only (halo, cache, best library), so every pair stays distinguishable
in a heatmap, where every colour sits next to every other. Which library won is written in the
cell. Each cell also carries text, so identity never depends on colour alone.

Needs pandas and matplotlib.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed: write files only
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Palette: validated categorical slots 1-3 (all pairs pass CVD and normal-vision checks),
# text in ink tokens, recessive surface. Diverging: blue <-> red through a neutral grey.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
EMPTY = "#e4e3df"
CATEGORIES = ["halo", "cache", "library"]
CATEGORY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
DIVERGING = LinearSegmentedColormap.from_list("cache_vs_halo", ["#c23b3a", "#f0efec", "#1c5cab"])

OURS = ("halo", "cache")
SHORT = {"halo": "halo", "cache": "cache", "cudnn": "cuDNN", "npp": "NPP", "opencv": "OCV"}


def read_bench_csv(path):
    """Return (dataframe, metadata dict) from a bench CSV with '# key: value' header lines."""
    meta = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = re.match(r"#\s*([a-z ]+):\s*(.*)", line.strip())
            if m:
                meta[m.group(1).strip()] = m.group(2).strip()
    df = pd.read_csv(path, comment="#")
    df["pixels"] = df["width"] * df["height"]
    df["size"] = df["width"].astype(str) + "x" + df["height"].astype(str)
    return df, meta


def device_line(meta, path):
    parts = [meta.get("label", ""), meta.get("device", "")]
    text = " | ".join(p for p in parts if p)
    return text or Path(path).stem


def axes_for(df):
    """Rows: filter sizes, small to large. Columns: input sizes ordered by pixel count."""
    ksizes = sorted(df["ksize"].unique())
    sizes = df.drop_duplicates("size").sort_values("pixels")["size"].tolist()
    return ksizes, sizes


def style_axes(ax, ksizes, sizes):
    ax.set_xticks(np.arange(len(sizes)))
    ax.set_xticklabels(sizes, rotation=60, ha="right", fontsize=7, color=INK_2)
    ax.set_yticks(np.arange(len(ksizes)))
    ax.set_yticklabels([f"{k}x{k}" for k in ksizes], fontsize=7, color=INK_2)
    ax.set_xlabel("input size (ordered by pixel count)", fontsize=8, color=INK_2)
    ax.set_ylabel("filter size", fontsize=8, color=INK_2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_cells(ax, values, cmap, norm=None):
    """pcolormesh with a surface-coloured gap between cells; NaN cells drawn as EMPTY."""
    cmap = cmap.copy() if hasattr(cmap, "copy") else cmap
    cmap.set_bad(EMPTY)
    rows, cols = values.shape
    return ax.pcolormesh(np.arange(cols + 1) - 0.5, np.arange(rows + 1) - 0.5,
                         np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                         edgecolors=SURFACE, linewidth=1.5)


def plot_winners(df, border, batch, title_device, out_path):
    d = df[(df["border"] == border) & (df["batch"] == batch) & (df["status"] == "ok")]
    if d.empty:
        return False
    ksizes, sizes = axes_for(df[df["batch"] == batch])
    cat = np.full((len(ksizes), len(sizes)), np.nan)
    text = [["" for _ in sizes] for _ in ksizes]

    for (k, s), cell in d.groupby(["ksize", "size"]):
        best = cell.loc[cell["median_ms"].idxmin()]
        runner = best["runner"]
        i, j = ksizes.index(k), sizes.index(s)
        cat[i, j] = CATEGORIES.index(runner) if runner in OURS else CATEGORIES.index("library")
        text[i][j] = SHORT.get(runner, runner)

    fig, ax = plt.subplots(figsize=(max(6, 0.62 * len(sizes) + 2), max(4, 0.36 * len(ksizes) + 2)),
                           facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    draw_cells(ax, cat, ListedColormap(CATEGORY_COLORS), norm=matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3))
    for i in range(len(ksizes)):
        for j in range(len(sizes)):
            label = text[i][j] if text[i][j] else "–"
            ax.text(j, i, label, ha="center", va="center", fontsize=6, color=INK)
    style_axes(ax, ksizes, sizes)
    ax.set_title(f"Fastest runner: border {border}, batch {batch}\n{title_device}",
                 fontsize=10, color=INK, loc="left")
    handles = [Patch(facecolor=c, edgecolor="none", label=l)
               for c, l in zip(CATEGORY_COLORS, ["halo (ours)", "cache (ours)", "library (named in cell)"])]
    handles.append(Patch(facecolor=EMPTY, edgecolor="none", label="no runner measured"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              fontsize=7, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_ratio(df, border, batch, title_device, out_path):
    d = df[(df["border"] == border) & (df["batch"] == batch) & (df["status"] == "ok")
           & (df["runner"].isin(OURS))]
    if d.empty:
        return False
    ksizes, sizes = axes_for(df[df["batch"] == batch])
    if "halo" not in d["runner"].values or "cache" not in d["runner"].values:
        return False
    halo = d[d["runner"] == "halo"].pivot_table(index="ksize", columns="size", values="median_ms")
    cache = d[d["runner"] == "cache"].pivot_table(index="ksize", columns="size", values="median_ms")
    ratio = (cache / halo).reindex(index=ksizes, columns=sizes)  # > 1: halo faster
    logr = np.log2(ratio.to_numpy(dtype=float))
    limit = max(1.0, float(np.nanmax(np.abs(logr)))) if np.isfinite(logr).any() else 1.0

    fig, ax = plt.subplots(figsize=(max(6, 0.62 * len(sizes) + 2.5), max(4, 0.36 * len(ksizes) + 2)),
                           facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    mesh = draw_cells(ax, logr, DIVERGING, norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
    for i in range(len(ksizes)):
        for j in range(len(sizes)):
            r = ratio.iat[i, j]
            if np.isfinite(r):
                label = f"{r:.1f}" if r >= 1 else f"{1 / r:.1f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=6, color=INK)
    style_axes(ax, ksizes, sizes)
    ax.set_title(f"Halo vs cache: times faster (blue = halo wins, red = cache wins)\n"
                 f"border {border}, batch {batch} | {title_device}", fontsize=10, color=INK, loc="left")
    bar = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.02)
    ticks = [t for t in range(-int(limit), int(limit) + 1)]
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"cache {2 ** -t:g}x" if t < 0 else ("equal" if t == 0 else f"halo {2 ** t:g}x")
                        for t in ticks])
    bar.ax.tick_params(labelsize=7, colors=INK_2, length=0)
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return True


def crossover_table(df):
    """First radius at which halo is faster than cache, per (border, batch, input size)."""
    d = df[(df["status"] == "ok") & (df["runner"].isin(OURS))]
    rows = []
    for (border, batch, size), cell in d.groupby(["border", "batch", "size"]):
        piv = cell.pivot_table(index="radius", columns="runner", values="median_ms")
        if not {"halo", "cache"} <= set(piv.columns):
            continue
        halo_wins = piv.index[piv["halo"] < piv["cache"]]
        first = int(halo_wins.min()) if len(halo_wins) else None
        rows.append({"border": border, "batch": batch, "size": size,
                     "pixels": int(cell["pixels"].iloc[0]),
                     "first_halo_win_radius": first,
                     "suggested_cache_max_radius": (first - 1) if first is not None else int(piv.index.max())})
    return pd.DataFrame(rows).sort_values(["border", "batch", "pixels"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="+", help="bench CSV file(s), one per device")
    ap.add_argument("--out", default="docs/assets", help="output folder (default docs/assets)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for path in args.csv:
        df, meta = read_bench_csv(path)
        tag = Path(path).stem
        title = device_line(meta, path)
        written = 0
        for border in sorted(df["border"].unique()):
            for batch in sorted(df["batch"].unique()):
                written += plot_winners(df, border, batch, title, out / f"winner_{tag}_{border}_b{batch}.png")
                written += plot_ratio(df, border, batch, title, out / f"ratio_{tag}_{border}_b{batch}.png")

        table = crossover_table(df)
        table.to_csv(out / f"crossover_{tag}.csv", index=False)

        status = df["status"].value_counts().to_dict()
        print(f"{path}: {len(df)} rows {status}; {written} figures -> {out}/")
        if not table.empty:
            summary = table.groupby("border")["suggested_cache_max_radius"].median()
            for border, value in summary.items():
                print(f"  {border}: median suggested cache_max_radius = {value:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())