"""
12_curate_gsea_dotplots.py

Hand-picks which GSEA terms appear on the dot plot, instead of taking the top N
by NES and getting "ear development". Run in two passes, template then replot.
The original GSEA results file is never modified.

Parameters
    MODE                  "template" first, then "replot"
    RANK_COL              "rho_global", which set of GSEA results to curate
    PREFILL_TOP_N         10 per direction, prefilled as "yes" in the template
    FDR_MAX               0.05, the gate for prefilling AND for what the
                          template contains at all
    OVERWRITE_TEMPLATE    False. True re-prefills and DISCARDS your edits
    DOTPLOT_CMAX          20, colour scale ceiling
    PANEL_W_MM, PANEL_H_MM
                          the axes rectangle, axis line to axis line, identical
                          on every panel so the figures overlay in Illustrator.
                          Margins are absolute, so the total canvas is fixed too
    NES_XLIM, NES_XTICKS_MAJOR, NES_MINOR_STEP
                          one x range for every group, closed with a tick at
                          each end
    Dot area uses fixed absolute size bins rather than percentiles of whatever
    is in the panel, so a set of a given size draws the same dot on every
    figure. The size key is written once as its own file rather than drawn on
    each panel.

Actions required
    Set MODE = "template" and run. For every
    <group>_<target>_<rank>_prerank_gsea_results.csv found, a sibling
    ..._toplot.csv is written with two editable columns at the front:
        plot            "yes" or "no", prefilled "yes" for the top N per
                        direction; only terms at FDR < FDR_MAX are written
        display_label   the text drawn on the figure, prefilled from Term with
                        the GO accession stripped
    Rows are ordered "yes" first, then "no", each block by NES descending.
    Edit that file in Excel. Change plot, rewrite display_label to whatever you
    want on the axis. Nothing else in it is read.
    Then set MODE = "replot" and run again.
        py 12_curate_gsea_dotplots.py

Input
    <RUN_DIR>/.../<group>_<target>_<rank>_prerank_gsea_results.csv
                                            from the point-level correlation
                                            script, which must have run first
    <RUN_DIR>/.../<group>_..._toplot.csv    your edited template, replot mode
Output
    ..._toplot.csv                          template mode
    4_<group>_pathway_dotplot_curated_<rank>_<target>.png and .svg
                                            replot mode, rows marked yes only,
                                            in NES order
    0_gene_set_size_legend.png and .svg     the size key, written once
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# =============================================================================
# CONFIG
# =============================================================================
MODE = "replot"            # "template" first, then "replot"

RUN_DIR = Path(
    r"Z:\Visium_Data\001_SpatialX\20260803_FinalAnnotatedMetadata"
    r"\pointlevel_expression_outputs_ageAdj"
)
TARGET = "Elasticity_l_Pa"
RANK_COL = "rho_global"

PREFILL_TOP_N = 10         # per direction, prefilled as "yes" in the template
FDR_MAX = 0.05
OVERWRITE_TEMPLATE = False  # True re-prefills and DISCARDS your edits

# Axes rectangle, axis line to axis line. Identical on every panel.
PANEL_W_MM = 26.862
PANEL_H_MM = 42.010

# Absolute margins around the panel. The canvas is the sum, so the axes sit at
# the same place in every file.
MARGIN_L_MM = 38.0         # pathway labels
MARGIN_R_MM = 15.0         # colourbar and its tick labels
MARGIN_B_MM = 8.0          # x ticks and axis label
MARGIN_T_MM = 6.0          # title
CBAR_W_MM = 2.5
CBAR_GAP_MM = 2.0

# One x range for every group, so the panels are comparable and overlayable.
NES_XLIM = (-3.0, 3.0)
NES_XTICKS_MAJOR = (-3.0, -2.0, 2.0, 3.0)
NES_MINOR_STEP = 0.5

# None fills the panel with however many terms are marked. An integer fixes the
# number of row slots, so row pitch is identical across panels and a panel with
# fewer terms leaves the remaining slots blank.
Y_SLOTS = None

DOTPLOT_CMAP = "plasma"
DOTPLOT_CMAX = 20
DOTPLOT_SIZE_LEGEND = False
LABEL_TRUNCATE = 35
PLOT_DPI = 600
BASE_FONT = 6

TERM_COL, NES_COL, FDR_COL, SIZE_COL = "Term", "NES", "FDR q-val", "Tag %"

MM = 1.0 / 25.4


# -----------------------------------------------------------------------------
# Gene-set size -> dot area, in FIXED bins.
#
# Percentile-derived sizes made the same 26-gene set look large in one panel and
# small in another, because the scale was rebuilt from whatever happened to be
# in that panel. These bins are absolute, so a dot of a given size means the
# same number of genes in the global plot and in every compartment plot.
#
# Sizes are declared as diameters in mm and converted to the points^2 that
# matplotlib's scatter expects, where s is the square of the diameter in
# points. Three discrete sizes only; nothing scales continuously.
# -----------------------------------------------------------------------------
DOT_DIAM_MM = (0.70, 1.50, 2.20)


def _area_pt2(d_mm):
    return (d_mm * 72.0 / 25.4) ** 2


GENE_SET_SIZE_BINS = [(20, _area_pt2(DOT_DIAM_MM[0]), "$\\leq$ 20"),
                      (50, _area_pt2(DOT_DIAM_MM[1]), "21 - 50"),
                      (float("inf"), _area_pt2(DOT_DIAM_MM[2]), "> 50")]


def gene_set_size_area(sizes):
    """Marker area (points^2) for each gene-set size, using the fixed bins."""
    s = pd.to_numeric(pd.Series(sizes).reset_index(drop=True), errors="coerce")
    out = np.full(len(s), GENE_SET_SIZE_BINS[-1][1], dtype=float)
    prev = -np.inf
    for hi, area, _ in GENE_SET_SIZE_BINS:
        out[(s > prev) & (s <= hi)] = area
        prev = hi
    out[s.isna()] = GENE_SET_SIZE_BINS[0][1]
    return out


def gene_set_size_handles(ax):
    """Three legend swatches, one per bin, identical on every panel."""
    return [ax.scatter([], [], s=area, facecolor="lightgrey",
                       edgecolor="black", linewidths=0.45, label=lab)
            for _, area, lab in GENE_SET_SIZE_BINS]


def save_gene_set_size_legend(out_dir, stem="0_gene_set_size_legend"):
    """
    The size key as its own small file.

    The bins are fixed and identical on every panel, so one legend serves the
    whole figure. Regenerate it whenever DOT_DIAM_MM changes, or
    it will state the wrong sizes.
    """
    fig, ax = plt.subplots(figsize=(1.3, 1.0))
    ax.set_axis_off()
    ax.legend(handles=gene_set_size_handles(ax), title="Genes in set",
              frameon=False, labelspacing=1.3, loc="center",
              fontsize=BASE_FONT, title_fontsize=BASE_FONT)
    for ext in ("png", "svg"):
        fig.savefig(Path(out_dir) / f"{stem}.{ext}",
                    dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f" Saved: {stem}.png / .svg  [gene-set size key]")


def clean_label(term):
    """Strip the GO accession and tidy the case for a figure axis."""
    t = re.sub(r"\s*\(GO:\d+\)\s*$", "", str(term)).strip()
    t = re.sub(r"\s*R-[A-Z]{3}-\d+\s*$", "", t).strip()
    return t


def find_results():
    pat = f"*_{TARGET}_{RANK_COL}_prerank_gsea_results.csv"
    return sorted(RUN_DIR.rglob(pat))


def group_of(path):
    m = re.match(rf"^(.*)_{re.escape(TARGET)}_{re.escape(RANK_COL)}"
                 rf"_prerank_gsea_results\.csv$", path.name)
    return m.group(1) if m else path.stem


# =============================================================================
# STEP 1: TEMPLATE
# =============================================================================
def write_template(f):
    out = f.with_name(f.name.replace("_prerank_gsea_results.csv", "_toplot.csv"))
    if out.exists() and not OVERWRITE_TEMPLATE:
        print(f"  [KEEP] {out.name} already exists; "
              f"set OVERWRITE_TEMPLATE = True to regenerate "
              f"(this DISCARDS your edits)")
        return

    d = pd.read_csv(f)
    for c in (TERM_COL, NES_COL, FDR_COL):
        if c not in d.columns:
            print(f"  [SKIP] {f.name}: no '{c}' column. Columns: "
                  f"{list(d.columns)}")
            return
    d[NES_COL] = pd.to_numeric(d[NES_COL], errors="coerce")
    d[FDR_COL] = pd.to_numeric(d[FDR_COL], errors="coerce")
    d = d.dropna(subset=[NES_COL, FDR_COL, TERM_COL])

    n_all = len(d)
    d = d[d[FDR_COL] < FDR_MAX].copy()
    if d.empty:
        print(f"  [SKIP] {f.name}: no terms at FDR < {FDR_MAX} "
              f"({n_all:,} tested)")
        return

    pre = set(d.nlargest(PREFILL_TOP_N, NES_COL)[TERM_COL]) | \
          set(d.nsmallest(PREFILL_TOP_N, NES_COL)[TERM_COL])

    d.insert(0, "display_label", d[TERM_COL].map(clean_label))
    d.insert(0, "plot", np.where(d[TERM_COL].isin(pre), "yes", "no"))
    d = d.sort_values(["plot", NES_COL], ascending=[False, False],
                      kind="mergesort").copy()

    d.to_csv(out, index=False)
    n_pos = int((d[NES_COL] > 0).sum())
    n_neg = int((d[NES_COL] < 0).sum())
    print(f"  {out.name}")
    print(f"     {len(d)} of {n_all:,} terms at FDR < {FDR_MAX} "
          f"({n_pos} positive, {n_neg} negative), yes first then NES "
          f"descending; {int((d['plot'] == 'yes').sum())} prefilled yes")


# =============================================================================
# STEP 2: REPLOT
# =============================================================================
def draw(d, stem, title, out_dir):
    """
    Exactly the rows given, in NES order. No re-filtering, no top-N.

    The axes rectangle is PANEL_W_MM x PANEL_H_MM on every panel and sits at a
    fixed offset from the canvas corner, so files from different groups
    register when stacked in Illustrator. Nothing about the data changes the
    geometry; long labels overrun the left margin rather than shrinking the
    panel, which is why LABEL_TRUNCATE exists.
    """
    d = d.sort_values(NES_COL).reset_index(drop=True)

    sz = pd.to_numeric(
        d[SIZE_COL].astype(str).str.extract(r"/\s*(\d+)")[0], errors="coerce") \
        if SIZE_COL in d.columns else pd.Series([np.nan] * len(d))
    if sz.isna().all() and SIZE_COL in d.columns:
        sz = pd.to_numeric(d[SIZE_COL].astype(str).str.replace("%", "",
                           regex=False), errors="coerce")
    if sz.isna().all():
        sz = pd.Series(np.full(len(d), 60.0))
    sz = sz.fillna(float(np.nanmedian(sz)))
    areas = gene_set_size_area(sz)

    off = d[(d[NES_COL] < NES_XLIM[0]) | (d[NES_COL] > NES_XLIM[1])]
    if len(off):
        print(f"     [WARN] {len(off)} term(s) fall outside NES "
              f"{NES_XLIM[0]} to {NES_XLIM[1]} and will not be visible: "
              + "; ".join(f"{t} ({v:.2f})" for t, v
                          in zip(off["display_label"], off[NES_COL])))

    nl = -np.log10(d[FDR_COL].clip(lower=1e-300))
    capped = nl >= DOTPLOT_CMAX
    labels = [f"{str(t)[:LABEL_TRUNCATE]}{' *' if c else ''}"
              for t, c in zip(d["display_label"], capped)]
    y = np.arange(len(d))

    n_slots = len(d) if Y_SLOTS is None else Y_SLOTS
    if Y_SLOTS is not None and len(d) > Y_SLOTS:
        print(f"     [WARN] {len(d)} terms exceed Y_SLOTS = {Y_SLOTS}; "
              f"rows above the slot count will sit outside the panel")

    fig_w_mm = MARGIN_L_MM + PANEL_W_MM + MARGIN_R_MM
    fig_h_mm = MARGIN_B_MM + PANEL_H_MM + MARGIN_T_MM
    fig = plt.figure(figsize=(fig_w_mm * MM, fig_h_mm * MM))
    ax = fig.add_axes([MARGIN_L_MM / fig_w_mm,
                       MARGIN_B_MM / fig_h_mm,
                       PANEL_W_MM / fig_w_mm,
                       PANEL_H_MM / fig_h_mm])
    cax = fig.add_axes([(MARGIN_L_MM + PANEL_W_MM + CBAR_GAP_MM) / fig_w_mm,
                        MARGIN_B_MM / fig_h_mm,
                        CBAR_W_MM / fig_w_mm,
                        PANEL_H_MM / fig_h_mm])

    ax.axvline(0, ls="--", color="grey", lw=0.4)
    sc = ax.scatter(d[NES_COL], y, c=nl.clip(upper=DOTPLOT_CMAX), s=areas,
                    cmap=DOTPLOT_CMAP, vmin=-np.log10(FDR_MAX),
                    vmax=DOTPLOT_CMAX, edgecolors="black", linewidths=0.25,
                    zorder=3)
    ax.set_ylim(-0.85, n_slots - 0.15)

    ax.set_xlim(*NES_XLIM)
    majors = [float(t) for t in NES_XTICKS_MAJOR]
    minors = [v for v in np.arange(NES_XLIM[0],
                                   NES_XLIM[1] + NES_MINOR_STEP / 2,
                                   NES_MINOR_STEP)
              if not any(abs(v - m) < 1e-9 for m in majors)]
    ax.xaxis.set_major_locator(mticker.FixedLocator(majors))
    ax.xaxis.set_minor_locator(mticker.FixedLocator(minors))
    ax.tick_params(axis="x", which="major", length=1.8, width=0.4,
                   pad=1.5, labelsize=BASE_FONT)
    ax.tick_params(axis="x", which="minor", length=1.0, width=0.35)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=BASE_FONT)
    ax.tick_params(axis="y", length=1.8, width=0.4, pad=1.5,
                   labelsize=BASE_FONT)
    ax.set_xlabel("NES", fontsize=BASE_FONT, labelpad=1.5)
    ax.set_title(title, fontsize=BASE_FONT, pad=2.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.4)

    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r"$-\log_{10}$FDR", fontsize=BASE_FONT, labelpad=1.5)
    cb.ax.tick_params(labelsize=BASE_FONT, length=1.5, width=0.4, pad=1.0)
    cb.outline.set_linewidth(0.4)

    if DOTPLOT_SIZE_LEGEND:
        ax.add_artist(ax.legend(handles=gene_set_size_handles(ax),
                                title="Genes in set", frameon=False,
                                labelspacing=1.1, loc="lower right",
                                fontsize=BASE_FONT,
                                title_fontsize=BASE_FONT))

    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}",
                    dpi=PLOT_DPI if ext == "png" else None, bbox_inches=None)
    plt.close(fig)
    print(f"     drawn: {stem}.png / .svg  ({len(d)} terms, "
          f"panel {PANEL_W_MM} x {PANEL_H_MM} mm, "
          f"canvas {fig_w_mm:.3f} x {fig_h_mm:.3f} mm, "
          f"row pitch {PANEL_H_MM / max(n_slots, 1):.2f} mm)")


def replot(f):
    tpl = f.with_name(f.name.replace("_prerank_gsea_results.csv", "_toplot.csv"))
    if not tpl.exists():
        print(f"  [SKIP] no {tpl.name}; run MODE = 'template' first")
        return
    d = pd.read_csv(tpl)
    if "plot" not in d.columns:
        print(f"  [SKIP] {tpl.name} has no 'plot' column")
        return
    keep = d["plot"].astype(str).str.strip().str.lower().isin(
        ["yes", "y", "true", "1"])
    d = d[keep].copy()
    if d.empty:
        print(f"  [SKIP] {tpl.name}: nothing marked yes")
        return
    d[NES_COL] = pd.to_numeric(d[NES_COL], errors="coerce")
    d[FDR_COL] = pd.to_numeric(d[FDR_COL], errors="coerce")
    d = d.dropna(subset=[NES_COL, FDR_COL])
    if "display_label" not in d.columns:
        d["display_label"] = d[TERM_COL].map(clean_label)
    d["display_label"] = d["display_label"].fillna("").astype(str)
    blank = d["display_label"].str.strip() == ""
    d.loc[blank, "display_label"] = d.loc[blank, TERM_COL].map(clean_label)

    n_above = int((d[FDR_COL] >= FDR_MAX).sum())
    if n_above:
        print(f"     [NOTE] {n_above} selected term(s) are ABOVE "
              f"FDR {FDR_MAX}; they will still be drawn because you chose them")

    g = group_of(f)
    draw(d, f"4_{g}_pathway_dotplot_curated_{RANK_COL}_{TARGET}",
         f"{g} \u00b7 curated GSEA ({RANK_COL})", f.parent)


# =============================================================================
def main():
    files = find_results()
    print(f"Run  : {RUN_DIR}")
    print(f"Mode : {MODE}")
    print(f"Found {len(files)} GSEA results file(s)\n")
    if not files:
        raise SystemExit("Nothing found. Check RUN_DIR, TARGET and RANK_COL.")
    if MODE == "replot" and not DOTPLOT_SIZE_LEGEND:
        save_gene_set_size_legend(RUN_DIR)

    n_terms = {}
    for f in files:
        print(f"{group_of(f)}")
        if MODE == "template":
            write_template(f)
        elif MODE == "replot":
            replot(f)
        else:
            raise SystemExit("MODE must be 'template' or 'replot'")
    if MODE == "template":
        print("\nEdit the _toplot.csv files, then set MODE = 'replot' "
              "and run again.")


if __name__ == "__main__":
    main()