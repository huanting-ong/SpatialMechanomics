# -*- coding: utf-8 -*-
"""
6_NI_stiffness_map.py

Draws the spatial stiffness heatmap for each region of interest, on a colour
scale shared across every sample so maps are directly comparable.

Parameters
    VALID_MIN, VALID_MAX    7.6 to 22,726 Pa, plausible range for E_l
    DISPLAY_VMIN, VMAX      0 to 3,000 Pa, the fixed colour scale
    GRID_STEP_UM            50, the acquisition grid pitch
    GRID_ORIGIN             1, the pipeline emits 1-based X and Y indices
    CELL_SIZE_IN            0.25, inches per grid square, so figure size
                            scales with grid dimensions and cells stay the
                            same physical size across maps
    UNIFORM_ALPHA           0.85
    DPI                     600
    Colormap YlOrRd.

Actions required
    Place in the project root, one folder per slice beneath it. Run script 1
    first.
        py 6_NI_stiffness_map.py                       all samples
        py 6_NI_stiffness_map.py --file "<path>"       one file
        py 6_NI_stiffness_map.py --xwin 0 4            restrict X indices

Input
    <root>/<slice>/output/NI_<slice>_all_samples.csv    from script 1
    Required columns: X, Y, Elasticity_l (Pa), Perr_l
Output
    <root>/<slice>/output/<slice>_Elasticity_l.png
    <root>/<slice>/output/<slice>_Elasticity_l.svg
    Both transparent. These are the panels in Fig. 2c,d and Supplementary
    Fig. 2.
"""

import os
import glob
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

# =============================================================================
# Configs
# =============================================================================
ROOT = os.path.dirname(__file__)+'/' # the filefolder which this py file is located
VALID_MIN, VALID_MAX = 7.6, 22726.0   # Pa
DISPLAY_VMIN = 0.0
DISPLAY_VMAX = 3000.0
GRID_STEP_UM = 50.0                   # index -> µm
UNIFORM_ALPHA = 0.85                  # 85%
DPI = 600
CELL_SIZE_IN = 0.25      # size of one grid square in inches
CBAR_WIDTH_IN = 0.2     # fixed colorbar width in inches
CBAR_PAD_IN = 0.15       # gap between map and colorbar in inches
MARGIN_LEFT_IN = 0.5
MARGIN_RIGHT_IN = 0.3
MARGIN_BOTTOM_IN = 0.5
MARGIN_TOP_IN = 0.4
GRID_ORIGIN = 1                       # pipeline emits 1-based X and Y indices

# =============================================================================

def get_stiffness_cmap():
    return plt.get_cmap("YlOrRd")

def tick_range(lo, hi, step):
    start = math.ceil(lo / step) * step
    end = math.floor(hi / step) * step
    return np.arange(start, end + 1e-6, step)

def plot_one_csv(input_path,
    x_index_window=None,
    valid_min=VALID_MIN, valid_max=VALID_MAX,
    grid_step_um=GRID_STEP_UM, uniform_alpha=UNIFORM_ALPHA,
    dpi=DPI, global_vmin=DISPLAY_VMIN, global_vmax=DISPLAY_VMAX):
    # parent name and output path
    output_dir = os.path.dirname(input_path)
    parent = os.path.basename(os.path.dirname(output_dir))  # <parent>/output/<file>
    out_path = os.path.join(output_dir, f"{parent}_Elasticity_l.png")

    # read
    df = pd.read_csv(input_path)
    df.columns = [c.strip() for c in df.columns]

    # required columns: index-based X, Y, and 'Elasticity_l (Pa)'
    for col in ["X", "Y", "Elasticity_l (Pa)"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {input_path}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ensure numeric
    df["Elasticity_l (Pa)"] = pd.to_numeric(df["Elasticity_l (Pa)"], errors="coerce")
    if "Perr_l" in df.columns:
        df["Perr_l"] = pd.to_numeric(df["Perr_l"], errors="coerce")
    else:
        raise ValueError(f"Missing 'Perr_l' column in {input_path}")

    # valid range filter (Elasticity_l) AND 15% Hertzian fit (Perr_l <= 0.15)
    mask_valid = (
        df["Elasticity_l (Pa)"].between(valid_min, valid_max, inclusive="both")
        & (df["Perr_l"] <= 0.15)
    )

    # optional X index window
    if x_index_window is not None:
        xmin_idx, xmax_idx = x_index_window
        mask_x = (df["X"] >= xmin_idx) & (df["X"] <= xmax_idx)
    else:
        mask_x = df["X"].notna()

    df_filt = df[mask_valid & mask_x].copy()

    # fixed grid cell size from index step
    dx = dy = grid_step_um
    half_dx, half_dy = dx/2.0, dy/2.0

    # derive positions from indices
    df_filt["X_um"] = (df_filt["X"] - GRID_ORIGIN) * grid_step_um + half_dx
    df_filt["Y_um"] = (df_filt["Y"] - GRID_ORIGIN) * grid_step_um + half_dy

    df_plot = df_filt[["X", "Y", "X_um", "Y_um", "Elasticity_l (Pa)"]].dropna()
    df_plot.rename(columns={"Elasticity_l (Pa)": "Elasticity"}, inplace=True)

    if df_plot.empty:
        print(f"[WARN] No data to plot after filters for: {input_path}")
        return None

    # values & colour mapping
    v = df_plot["Elasticity"].astype(float).values

    vmin, vmax = float(global_vmin), float(global_vmax)

    cmap = get_stiffness_cmap()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    # limits
    if x_index_window is not None:
        xmin = (xmin_idx - GRID_ORIGIN) * grid_step_um
        xmax = (xmax_idx - GRID_ORIGIN + 1) * grid_step_um
    else:
        xmin = (df_plot["X"].min() - GRID_ORIGIN) * grid_step_um
        xmax = (df_plot["X"].max() - GRID_ORIGIN + 1) * grid_step_um

    ymin = (df_plot["Y"].min() - GRID_ORIGIN) * grid_step_um
    ymax = (df_plot["Y"].max() - GRID_ORIGIN + 1) * grid_step_um

    # figure size based on grid dimensions, with fixed cell size and fixed colorbar size
    if x_index_window is not None:
        n_cols = int(xmax_idx - xmin_idx + 1)
    else:
        n_cols = int(df_plot["X"].max() - df_plot["X"].min() + 1)

    n_rows = int(df_plot["Y"].max() - df_plot["Y"].min() + 1)

    map_width_in = max(1, n_cols * CELL_SIZE_IN)
    map_height_in = max(1, n_rows * CELL_SIZE_IN)

    fig_width_in = MARGIN_LEFT_IN + map_width_in + CBAR_PAD_IN + CBAR_WIDTH_IN + MARGIN_RIGHT_IN
    fig_height_in = MARGIN_BOTTOM_IN + map_height_in + MARGIN_TOP_IN

    fig = plt.figure(figsize=(fig_width_in, fig_height_in), facecolor="none")
    ax = fig.add_axes([
        MARGIN_LEFT_IN / fig_width_in,
        MARGIN_BOTTOM_IN / fig_height_in,
        map_width_in / fig_width_in,
        map_height_in / fig_height_in
    ])
    ax.set_facecolor("none")

    for (x, y, val) in zip(df_plot["X_um"], df_plot["Y_um"], v):
        val_plot = np.clip(val, vmin, vmax)
        rgba = list(sm.to_rgba(val_plot))
        rgba[3] = float(uniform_alpha)  # uniform alpha across all valid cells
        rect = Rectangle((x - half_dx, y - half_dy), dx, dy,
                         facecolor=tuple(rgba), edgecolor=(0, 0, 0, 0))
        ax.add_patch(rect)

    # axes style (white outward ticks; transparent bg)
    ax.set_xlabel("X (µm)", color="black", fontsize=6, fontname="Arial")
    ax.set_ylabel("Y (µm)", color="black", fontsize=6, fontname="Arial")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.tick_params(colors="black", labelsize=6, direction="out", length=2)
    
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontname("Arial")
        lab.set_fontsize(6)

    ax.grid(False)
    ax.set_aspect("equal")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # ticks every 50 µm, labels every 100 µm
    minor_step_um = 50.0
    major_step_um = 100.0

    xticks_all = np.arange(xmin, xmax + 1e-6, minor_step_um)
    yticks_all = np.arange(ymin, ymax + 1e-6, minor_step_um)

    ax.set_xticks(xticks_all)
    ax.set_yticks(yticks_all)

    xlabels = [str(int(t)) if (int(round(t)) % int(major_step_um) == 0) else "" for t in xticks_all]
    ylabels = [str(int(t)) if (int(round(t)) % int(major_step_um) == 0) else "" for t in yticks_all]

    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontname("Arial")
        lab.set_fontsize(6)
        lab.set_color("black")

    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")

    # colorbar (fixed width and aligned to the map height)
    cax = fig.add_axes([
        (MARGIN_LEFT_IN + map_width_in + CBAR_PAD_IN) / fig_width_in,
        MARGIN_BOTTOM_IN / fig_height_in,
        CBAR_WIDTH_IN / fig_width_in,
        map_height_in / fig_height_in
    ])

    cbar = fig.colorbar(sm, cax=cax)

    ticks = [0, 750, 1500, 2250, 3000]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])

    cbar.ax.tick_params(colors="black", labelsize=6, direction="out", length=2)
    for lab in cbar.ax.get_yticklabels():
        lab.set_fontname("Arial")
        lab.set_fontsize(6)

    cbar.ax.set_title(r"$E$ (Pa)", color="black", fontsize=6, pad=6, fontname="Arial")

    # save
    out_path_png = os.path.join(output_dir, f"{parent}_Elasticity_l.png")
    out_path_svg = os.path.join(output_dir, f"{parent}_Elasticity_l.svg")

    fig.savefig(out_path_png, dpi=dpi, transparent=True)
    fig.savefig(out_path_svg, transparent=True)

    plt.close(fig)
    print(f"[OK] Wrote: {out_path_png}")
    print(f"[OK] Wrote: {out_path_svg}")
    return out_path_png

def find_all_inputs(root=ROOT):
    parents = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    files = [os.path.join(d, "output", f"NI_{os.path.basename(d)}_all_samples.csv")
             for d in parents]
    return sorted(f for f in files if os.path.isfile(f))

def collect_global_elasticity_range(files, x_index_window=None,
                                    valid_min=VALID_MIN, valid_max=VALID_MAX):
    vals = []

    for input_path in files:
        try:
            df = pd.read_csv(input_path)
            df.columns = [c.strip() for c in df.columns]

            for col in ["X", "Y", "Elasticity_l (Pa)"]:
                if col not in df.columns:
                    raise ValueError(f"Missing required column '{col}' in {input_path}")
                df[col] = pd.to_numeric(df[col], errors="coerce")

            if "Perr_l" not in df.columns:
                raise ValueError(f"Missing 'Perr_l' column in {input_path}")
            df["Perr_l"] = pd.to_numeric(df["Perr_l"], errors="coerce")

            mask_valid = (
                df["Elasticity_l (Pa)"].between(valid_min, valid_max, inclusive="both")
                & (df["Perr_l"] <= 0.15)
            )

            if x_index_window is not None:
                xmin_idx, xmax_idx = x_index_window
                mask_x = (df["X"] >= xmin_idx) & (df["X"] <= xmax_idx)
            else:
                mask_x = df["X"].notna()

            v = df.loc[mask_valid & mask_x, "Elasticity_l (Pa)"].dropna().astype(float).values
            if len(v) > 0:
                vals.append(v)

        except Exception as e:
            print(f"[WARN] Skipping range collection for {input_path}: {e}")

    if not vals:
        raise ValueError("No valid Elasticity_l values found across input files.")

    all_vals = np.concatenate(vals)
    global_vmin = float(np.nanpercentile(all_vals, 5))
    global_vmax = float(np.nanpercentile(all_vals, 95))

    print(f"[INFO] Global Elasticity_l scale from all valid points:")
    print(f"[INFO] 5th percentile = {global_vmin:.3f} Pa")
    print(f"[INFO] 95th percentile = {global_vmax:.3f} Pa")

    return global_vmin, global_vmax

def main():
    ap = argparse.ArgumentParser(description="Generate stiffness maps (Elasticity_l) in default style.")
    ap.add_argument("--file", type=str, default=None,
                    help="Path to a single input CSV (e.g., Ting-Nanoindentation2025_correctedGeoFactor/<parent>/output/all samples.csv).")
    ap.add_argument("--xwin", nargs=2, type=int, default=None,
                    help="X index window as two ints, e.g., --xwin 0 4 for 0..200 µm.")
    args = ap.parse_args()

    xwin = tuple(args.xwin) if args.xwin is not None else None

    if args.file:
        if not os.path.isfile(args.file):
            print(f"[ERR] File not found: {args.file}")
            return
        files = [args.file]
    else:
        files = find_all_inputs(ROOT)
        if not files:
            print("[INFO] No input CSVs found.")
            return

    global_vmin, global_vmax = DISPLAY_VMIN, DISPLAY_VMAX

    for f in files:
        try:
            plot_one_csv(
                f,
                x_index_window=xwin,
                global_vmin=global_vmin,
                global_vmax=global_vmax
            )
        except Exception as e:
            print(f"[ERR] {f}: {e}")

if __name__ == "__main__":
    main()
