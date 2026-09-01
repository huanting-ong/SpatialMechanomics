"""
11_curate_gsea_dotplots.py

Hand-picks which GSEA terms appear on the dot plot, instead of taking the top N
by NES and getting "ear development". Run in two passes, template then replot.
The original GSEA results file is never modified.

Parameters
    MODE                  "template" first, then "replot"
    RANK_COL              "rho_global", which set of GSEA results to curate
    PREFILL_TOP_N         10 per direction, prefilled as "yes" in the template
    FDR_MAX               0.05, the gate for prefilling
    OVERWRITE_TEMPLATE    False. True re-prefills and DISCARDS your edits
    DOTPLOT_CMAX          20, colour scale ceiling
    Dot area uses fixed absolute size bins rather than percentiles of whatever
    is in the panel, so a set of a given size draws the same dot on every
    figure. The size key is written once as its own file rather than drawn on
    each panel.

Actions required
    Set MODE = "template" and run. For every
    <group>_<target>_<rank>_prerank_gsea_results.csv found, a sibling
    ..._toplot.csv is written with two editable columns at the front:
        plot            "yes" or "no", prefilled "yes" for the top N per
                        direction among FDR < FDR_MAX
        display_label   the text drawn on the figure, prefilled from Term with
                        the GO accession stripped
    Edit that file in Excel. Change plot, rewrite display_label to whatever you
    want on the axis. Nothing else in it is read.
    Then set MODE = "replot" and run again.
        py 11_curate_gsea_dotplots.py

Input
    <RUN_DIR>/.../<group>_<target>_<rank>_prerank_gsea_results.csv
                                            from script 10
    <RUN_DIR>/.../<group>_..._toplot.csv    your edited template, replot mode
Output
    ..._toplot.csv                          template mode
    4_<group>_pathway_dotplot_curated_<rank>_<target>.png and .svg
                                            replot mode, rows marked yes only,
                                            in NES order. Styling matches the
                                            automatic dot plot, so the two are
                                            interchangeable
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================
MODE = "template"          # "template" first, then "replot"

RUN_DIR = Path(
    r"Z:\Visium_Data\001_SpatialX\20260803_FinalAnnotatedMetadata"
    r"\pointlevel_expression_outputs_ageAdj"
)
TARGET = "Elasticity_l_Pa"
RANK_COL = "rho_global"

PREFILL_TOP_N = 10         # per direction, prefilled as "yes" in the template
FDR_MAX = 0.05
OVERWRITE_TEMPLATE = False  # True re-prefills and DISCARDS your edits

# Figure, matching the automatic dot plot
DOTPLOT_W_MM = 120.0
DOTPLOT_ROW_MM = 4.0
DOTPLOT_PAD_MM = 12.0
DOTPLOT_CMAX = 20
# The size bins are fixed and identical on every panel, so the key is
# written once as its own file rather than drawn over the data on each.
DOTPLOT_SIZE_LEGEND = False
LABEL_TRUNCATE = 58
PLOT_DPI = 600
BASE_FONT = 6

TERM_COL, NES_COL, FDR_COL, SIZE_COL = "Term", "NES", "FDR q-val", "Tag %"


# -----------------------------------------------------------------------------
# Gene-set size -> dot area, in FIXED bins.
#
# Percentile-derived sizes made the same 26-gene set look large in one panel and
# small in another, because the scale was rebuilt from whatever happened to be
# in that panel. These bins are absolute, so a dot of a given size means the
# same number of genes in the global plot and in every compartment plot.
# -----------------------------------------------------------------------------
GENE_SET_SIZE_BINS = [(20, 14.0, "$\\leq$ 20"),
                      (50, 45.0, "21 - 50"),
                      (float("inf"), 110.0, "> 50")]


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
    whole figure. On a 120 x 36 mm panel an in-axes legend sits on top of the
    data, which is what it was doing.
    """
    fig, ax = plt.subplots(figsize=(1.3, 1.0))
    ax.set_axis_off()
    ax.legend(handles=gene_set_size_handles(ax), title="Genes in set",
              frameon=False, labelspacing=1.3, loc="center",
              fontsize=6, title_fontsize=6)
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

    sig = d[d[FDR_COL] < FDR_MAX]
    pre = set(sig.nlargest(PREFILL_TOP_N, NES_COL)[TERM_COL]) | \
          set(sig.nsmallest(PREFILL_TOP_N, NES_COL)[TERM_COL])

    d = d.sort_values([FDR_COL, NES_COL], ascending=[True, False]).copy()
    d.insert(0, "display_label", d[TERM_COL].map(clean_label))
    d.insert(0, "plot", np.where(d[TERM_COL].isin(pre), "yes", "no"))

    d.to_csv(out, index=False)
    n_pos = int((sig[NES_COL] > 0).sum())
    n_neg = int((sig[NES_COL] < 0).sum())
    print(f"  {out.name}")
    print(f"     {len(d):,} terms, {len(sig)} at FDR < {FDR_MAX} "
          f"({n_pos} positive, {n_neg} negative); "
          f"{int((d['plot'] == 'yes').sum())} prefilled yes")


# =============================================================================
# STEP 2: REPLOT
# =============================================================================
def draw(d, stem, title, out_dir):
    """Exactly the rows given, in NES order. No re-filtering, no top-N."""
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

    nl = -np.log10(d[FDR_COL].clip(lower=1e-300))
    capped = nl >= DOTPLOT_CMAX
    labels = [f"{str(t)[:LABEL_TRUNCATE]}{' *' if c else ''}"
              for t, c in zip(d["display_label"], capped)]
    y = np.arange(len(d))

    fig_h = (DOTPLOT_PAD_MM + DOTPLOT_ROW_MM * len(d)) / 25.4
    fig, ax = plt.subplots(figsize=(DOTPLOT_W_MM / 25.4, fig_h))
    fig.set_layout_engine("constrained")
    if float(d[NES_COL].min()) < 0:
        ax.axvline(0, ls="--", color="grey", lw=0.6)
    sc = ax.scatter(d[NES_COL], y, c=nl.clip(upper=DOTPLOT_CMAX), s=areas,
                    cmap="Reds", vmin=-np.log10(FDR_MAX), vmax=DOTPLOT_CMAX,
                    edgecolors="black", linewidths=0.45, zorder=3)
    # Autoscale ignores marker radius, so the largest dot on the top row was
    # being sliced by the axes edge. Pad by most of a row, and a little on x.
    ax.set_ylim(-0.85, len(d) - 0.15)
    _xmin, _xmax = float(d[NES_COL].min()), float(d[NES_COL].max())
    _pad = max((_xmax - _xmin) * 0.08, 0.15)
    ax.set_xlim(_xmin - _pad, _xmax + _pad)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=BASE_FONT)
    ax.set_xlabel("NES", fontsize=BASE_FONT)
    ax.set_title(title, fontsize=BASE_FONT)
    ax.tick_params(labelsize=BASE_FONT)
    ax.spines[["top", "right"]].set_visible(False)

    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$-\log_{10}$FDR", fontsize=BASE_FONT)
    cb.ax.tick_params(labelsize=BASE_FONT)

    _draw_size_legend = DOTPLOT_SIZE_LEGEND
    if _draw_size_legend:
        ax.add_artist(ax.legend(handles=gene_set_size_handles(ax), title="Genes in set", frameon=False,
                                    labelspacing=1.1, loc="lower right",
                                    fontsize=BASE_FONT, title_fontsize=BASE_FONT))

    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}",
                    dpi=PLOT_DPI if ext == "png" else None, bbox_inches=None)
    plt.close(fig)
    print(f"     drawn: {stem}.png / .svg  ({len(d)} terms)")


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