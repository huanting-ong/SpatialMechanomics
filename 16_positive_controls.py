"""
16_positive_controls.py

Four controls that test whether the nanoindentation-to-transcriptome
registration is sound, so that a null result from script 12 can be attributed
to biology rather than to misalignment.

Why
    A null has two explanations that demand opposite responses. Either local
    stiffness genuinely does not predict local transcription once depth and
    animal are controlled, in which case the finding is real, or the
    registration is not accurate enough and each measurement is paired with the
    wrong transcriptome, in which case the null says nothing. The null itself
    cannot distinguish them. What is needed is something that must be true if
    the registration works.

The controls
    A   Spatial autocorrelation of stiffness. Neighbouring measurements sit
        tens of microns apart in the same tissue, so their stiffness must be
        correlated. Uses the mechanical data alone, no transcriptome
    B   Stiffness by compartment. Corpus luteum, follicle, stroma and
        vasculature are mechanically distinct, so stiffness must separate them
        if the values are landing on the structures the annotation says they
        are. Also tested within animal, so it cannot be driven by one mouse
    C   Composition against stiffness. Per measurement, the fraction of bins in
        each compartment and each cell cluster. Since a measurement aggregates
        about 92 bins, asking whether expression tracks stiffness is largely a
        question about which cells are present, so this sets the ceiling
        against which single-gene associations should be read
    D   Animal-level association at n = 8. Median stiffness per ovary against
        ovary-level pseudobulk. Eight is a small number of observations and the
        script says so; it is reported as a scale check, not as a test

Reading the result
    A and B pass            registration is sound, a null is biological
    A fails                 the mechanical map is noise, fix upstream
    A passes, B fails       the grid is internally consistent but sits in the
                            wrong place relative to the tissue

Actions required
    Set METADATA_TSV. Run script 9 first.
        py 16_positive_controls.py

Input
    metadata_with_nano.tsv                  from script 9
    expression_export/                      from script 10, control D only
Output
    Control tables and figures in their own folder. Evidence for
    Supplementary Methods S11.

Note
    Control B is age-dependent. Compartment representation differs strongly
    between young and aged, with corpus luteum making up 61% of young
    measurements and 5% of aged, so the pooled comparison partly reflects age.
    The within-animal test removes that confound and is the version to read.
    The age-stratified mixed model in script 5 supersedes both for any
    biological claim about compartments.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io as sio
from scipy import sparse
from scipy.stats import spearmanr, kruskal, rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================

METADATA_TSV = Path(
    r"Z:\Visium_Data\001_SpatialX\20260803_FinalAnnotatedMetadata\metadata_with_nano.tsv"
)
EXPR_DIR = METADATA_TSV.parent / "expression_export"
OUT_DIR = METADATA_TSV.parent / "positive_controls"

TARGET_COL = "Elasticity_l_Pa"
LOG_TARGET = True

POINT_KEY_COLS = ["Tissue.slices", "ROIs", "nano_point_id"]
NANO_ID_COL = "nano_point_id"
COMPARTMENT_COL = "Compartments"
CLUSTER_COL = "CellCluster"
AGE_COL = "AgeGroup"
TISSUE_COL = "Tissue.slices"
ROI_COL = "ROIs"
COL_X, COL_Y = "x", "y"

MIN_BINS_PER_POINT = 5
K_NEIGHBOURS = 4
SCALE_FACTOR = 1e4

COMP_COLOURS = {"F": "#af5420", "CL": "#8e6bd0", "S": "#54a397", "V": "#b44769"}
YOUNG_COL, AGED_COL = "#007A7A", "#B57BEE"
PLOT_DPI = 600

EXPR_MATRIX_FILE = "step9_expression_matrix.mtx"
EXPR_BARCODES_FILE = "step9_expression_barcodes.txt"
EXPR_FEATURES_FILE = "step9_expression_features.txt"
EXPR_MATCHED_METADATA_FILE = "step9_expression_metadata_matched.tsv"


def clean_bool(s):
    return s.astype(str).str.strip().str.lower() == "true"


def partial_spearman(x, y, covars):
    n = len(x)
    if covars is None:
        r = spearmanr(x, y)
        return float(r.correlation), float(r.pvalue), n
    k = covars.shape[1]
    if n < k + 10:
        return np.nan, np.nan, n
    rx, ry = rankdata(x).astype(float), rankdata(y).astype(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan, np.nan, n
    Z = np.column_stack([np.ones(n)] +
                        [rankdata(covars[:, j]).astype(float) for j in range(k)])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    if np.std(ex) == 0 or np.std(ey) == 0:
        return np.nan, np.nan, n
    from scipy.stats import t as student_t
    r = float(np.corrcoef(ex, ey)[0, 1])
    df = n - 2 - k
    if df <= 0 or abs(r) >= 1:
        return r, np.nan, n
    tt = r * np.sqrt(df / (1 - r ** 2))
    return r, float(2 * student_t.sf(abs(tt), df)), n


# =============================================================================
# LOAD AND AGGREGATE TO POINTS
# =============================================================================

def build_points():
    with open(EXPR_DIR / EXPR_BARCODES_FILE, encoding="utf-8") as f:
        barcodes = np.array([l.strip() for l in f])
    with open(EXPR_DIR / EXPR_FEATURES_FILE, encoding="utf-8") as f:
        genes = np.array([l.strip() for l in f])
    X = sio.mmread(EXPR_DIR / EXPR_MATRIX_FILE).tocsr()
    if X.shape[0] == len(genes) and X.shape[1] == len(barcodes):
        X = X.T.tocsr()

    meta = pd.read_csv(EXPR_DIR / EXPR_MATCHED_METADATA_FILE,
                       sep="\t", low_memory=False)
    if len(meta) != X.shape[0]:
        raise ValueError("metadata and matrix are out of sync")

    # reconstruct counts
    n_count = pd.to_numeric(meta["nCount_RNA"], errors="coerce").values.astype(float)
    R = sparse.csr_matrix(X, copy=True)
    R.data = np.expm1(R.data)
    R.data *= np.repeat(n_count, np.diff(R.indptr))
    unit = R.data.min()
    R.data = np.round(R.data / unit)
    R.eliminate_zeros()
    X = R

    meta[TARGET_COL] = pd.to_numeric(meta[TARGET_COL], errors="coerce")
    # .to_numpy(copy=True): under pandas copy-on-write, Series.values can be a
    # read-only view, and the in-place &= below then raises
    # "output array is read-only".
    keep = meta[TARGET_COL].notna().to_numpy(copy=True)
    if "valid_measurement" in meta.columns:
        keep &= clean_bool(meta["valid_measurement"]).to_numpy()
    keep &= meta[NANO_ID_COL].notna().to_numpy()
    keep &= meta[NANO_ID_COL].astype(str).str.strip().ne("N/A").to_numpy()
    X, meta = X[keep], meta.loc[keep].reset_index(drop=True)

    key_cols = [c for c in POINT_KEY_COLS if c in meta.columns]
    pid = meta[key_cols].astype(str).agg("||".join, axis=1).values
    uniq, inv = np.unique(pid, return_inverse=True)
    nb = np.bincount(inv).astype(float)

    ind = sparse.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))),
                            shape=(len(uniq), len(inv)))
    agg = (ind @ X).tocsr()

    def maj(col):
        if col not in meta.columns:
            return np.full(len(uniq), "Unknown", dtype=object)
        return (pd.Series(meta[col].values).groupby(inv)
                .agg(lambda v: v.dropna().astype(str).value_counts().idxmax()
                     if v.notna().any() else "Unknown")
                .reindex(range(len(uniq))).values)

    tgt = meta[TARGET_COL].values
    pm = pd.DataFrame({
        "point_id": uniq,
        "n_bins": nb,
        "target": np.bincount(inv, weights=tgt) / nb,
        "x": np.bincount(inv, weights=meta[COL_X].values) / nb,
        "y": np.bincount(inv, weights=meta[COL_Y].values) / nb,
        "compartment": maj(COMPARTMENT_COL),
        "cluster": maj(CLUSTER_COL),
        "slice": maj(TISSUE_COL),
        "roi": maj(ROI_COL),
    })
    pm["animal"] = pm["slice"].astype(str).str.replace(r"-\d+$", "", regex=True)
    pm["age_group"] = np.where(pm["animal"].str.startswith("Y"), "young", "aged")

    # per-point composition fractions
    comp_frac = (pd.crosstab(inv, meta[COMPARTMENT_COL].astype(str), normalize="index")
                 .reindex(range(len(uniq))).fillna(0))
    clus_frac = (pd.crosstab(inv, meta[CLUSTER_COL].astype(str), normalize="index")
                 .reindex(range(len(uniq))).fillna(0))

    ok = nb >= MIN_BINS_PER_POINT
    return (agg[ok], pm.loc[ok].reset_index(drop=True), genes,
            comp_frac.loc[ok].reset_index(drop=True),
            clus_frac.loc[ok].reset_index(drop=True))


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agg, pm, genes, comp_frac, clus_frac = build_points()
    print(f"points: {len(pm):,}   animals: {pm['animal'].nunique()}   "
          f"slices: {pm['slice'].nunique()}")

    y = pm["target"].values.astype(float)
    if LOG_TARGET and TARGET_COL in {"Elasticity_l_Pa", "Elasticity_eff_Pa"}:
        y = np.log10(y)
    animals = pm["animal"].values
    lev = np.unique(animals)
    A = np.column_stack([(animals == a).astype(float) for a in lev[1:]])

    # =========================================================================
    # A. Is the stiffness map spatially coherent at all?
    # =========================================================================
    print("\n" + "=" * 72)
    print("A. SPATIAL AUTOCORRELATION OF STIFFNESS  (no transcriptome involved)")
    print("=" * 72)
    print("  A point's stiffness vs the mean of its", K_NEIGHBOURS,
          "nearest neighbours,")
    print("  computed within each slice so neighbours are genuinely adjacent.")

    rows = []
    own_all, nbr_all = [], []
    for sl, idx in pm.groupby("slice").groups.items():
        idx = np.asarray(idx)
        if len(idx) < K_NEIGHBOURS + 3:
            continue
        P = pm.loc[idx, ["x", "y"]].values.astype(float)
        v = y[idx]
        d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn = np.argsort(d2, axis=1)[:, :K_NEIGHBOURS]
        nbr = v[nn].mean(axis=1)
        r, p = spearmanr(v, nbr)
        med_d = float(np.median(np.sqrt(np.take_along_axis(d2, nn, 1))))
        rows.append(dict(slice=sl, n_points=len(idx), rho=r, pval=p,
                         median_nn_distance=med_d))
        own_all.append(v)
        nbr_all.append(nbr)

    sp = pd.DataFrame(rows).sort_values("rho", ascending=False)
    sp.to_csv(OUT_DIR / "A_spatial_autocorrelation.csv", index=False)
    print(sp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    own_all, nbr_all = np.concatenate(own_all), np.concatenate(nbr_all)
    r_pool, p_pool = spearmanr(own_all, nbr_all)
    print(f"\n  pooled across slices: rho = {r_pool:+.3f}  p = {p_pool:.3g}  "
          f"n = {len(own_all):,}")
    print(f"  slices with rho > 0.3: {int((sp['rho'] > 0.3).sum())} of {len(sp)}")
    if r_pool < 0.2:
        print("  [FAIL] The stiffness map is not spatially coherent. Neighbouring")
        print("         indentations disagree. Nothing downstream can work until")
        print("         this is understood. Suspect the grid indexing or the")
        print("         contact/quality filters, not the transcriptome.")
    else:
        print("  [PASS] Stiffness is spatially structured, so the NI map itself")
        print("         is sound.")

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(own_all, nbr_all, s=5, alpha=0.25, color="grey",
               linewidths=0, rasterized=True)
    ax.set_xlabel(f"log10 {TARGET_COL}, this point")
    ax.set_ylabel(f"log10 {TARGET_COL}, mean of {K_NEIGHBOURS} neighbours")
    ax.set_title(f"Spatial coherence: rho = {r_pool:+.3f}", fontsize=9,
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(OUT_DIR / f"A_spatial_autocorrelation.{e}",
                    dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # B. Does stiffness separate the compartments?
    # =========================================================================
    print("\n" + "=" * 72)
    print("B. STIFFNESS BY COMPARTMENT")
    print("=" * 72)

    comps = [c for c in ["F", "CL", "S", "V"] if (pm["compartment"] == c).any()]
    groups = [y[pm["compartment"].values == c] for c in comps]
    if len(groups) >= 2:
        H, p = kruskal(*groups)
        print(f"  Kruskal-Wallis across {len(comps)} compartments: "
              f"H = {H:.1f}, p = {p:.3g}")
        for c, g in zip(comps, groups):
            print(f"    {c:>3s}  n = {len(g):5,}   median log10 E = "
                  f"{np.median(g):+.3f}   (E = {10 ** np.median(g):.0f} Pa)")

    # within-animal version: does the ordering hold inside each mouse?
    print("\n  Within each animal (so no single mouse can drive it):")
    per_animal = []
    for a in lev:
        m = animals == a
        gs = [y[m & (pm['compartment'].values == c)] for c in comps]
        gs_ok = [g for g in gs if len(g) >= 10]
        if len(gs_ok) < 2:
            continue
        H, p = kruskal(*gs_ok)
        meds = {c: (float(np.median(g)) if len(g) >= 10 else np.nan)
                for c, g in zip(comps, gs)}
        per_animal.append(dict(animal=a, H=H, pval=p, **meds))
    pa = pd.DataFrame(per_animal)
    if len(pa):
        pa.to_csv(OUT_DIR / "B_stiffness_by_compartment_per_animal.csv", index=False)
        print(pa.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        n_sig = int((pa["pval"] < 0.05).sum())
        print(f"\n  animals where compartments differ (p < 0.05): "
              f"{n_sig} of {len(pa)}")
        if n_sig < 0.5 * len(pa):
            print("  [FAIL] Stiffness does not distinguish tissue compartments")
            print("         within animals. If the NI map is spatially coherent")
            print("         (control A passed) but does not align with the")
            print("         histology, the NI and Visium coordinate frames are")
            print("         offset. Re-check 7_Corr_roi_bin_assignment.py.")
        else:
            print("  [PASS] Stiffness tracks tissue architecture, so the")
            print("         NI-to-bin registration is landing correctly.")

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    data = [y[pm["compartment"].values == c] for c in comps]
    bp = ax.boxplot(data, tick_labels=comps, patch_artist=True, showfliers=False,
                    widths=0.6)
    for patch, c in zip(bp["boxes"], comps):
        patch.set_facecolor(COMP_COLOURS.get(c, "grey"))
        patch.set_alpha(0.65)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_ylabel(f"log10 {TARGET_COL}")
    ax.set_xlabel("compartment")
    ax.set_title("Positive control: stiffness by compartment",
                 fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for e in ("png", "svg"):
        fig.savefig(OUT_DIR / f"B_stiffness_by_compartment.{e}",
                    dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # C. Does CELL COMPOSITION track stiffness?
    # =========================================================================
    print("\n" + "=" * 72)
    print("C. COMPOSITION vs STIFFNESS  (animal held constant)")
    print("=" * 72)
    print("  Each point is ~92 bins, so 'expression tracks stiffness' is mostly")
    print("  a question about which cells are present. This asks that directly.")

    rows = []
    for label, frame in [("compartment", comp_frac), ("cluster", clus_frac)]:
        for col in frame.columns:
            v = frame[col].values.astype(float)
            if np.std(v) == 0 or (v > 0).mean() < 0.05:
                continue
            r_raw, p_raw, _ = partial_spearman(v, y, None)
            r_w, p_w, n = partial_spearman(v, y, A)
            rows.append(dict(level=label, feature=str(col),
                             mean_fraction=float(v.mean()),
                             rho_raw=r_raw, p_raw=p_raw,
                             rho_within_animal=r_w, p_within_animal=p_w, n=n))
    comp_res = pd.DataFrame(rows)
    if len(comp_res):
        from statsmodels.stats.multitest import multipletests
        m = comp_res["p_within_animal"].notna()
        comp_res.loc[m, "fdr_within_animal"] = multipletests(
            comp_res.loc[m, "p_within_animal"], method="fdr_bh")[1]
        comp_res = comp_res.sort_values("rho_within_animal",
                                        key=lambda s: -s.abs())
        comp_res.to_csv(OUT_DIR / "C_composition_vs_stiffness.csv", index=False)
        print(comp_res.head(20).to_string(index=False,
                                          float_format=lambda v: f"{v:.3f}"))
        n_hit = int((comp_res["fdr_within_animal"] < 0.05).sum())
        print(f"\n  cell types whose abundance tracks stiffness WITHIN animals "
              f"(FDR < 0.05): {n_hit}")
        if n_hit:
            print("  This is a real, interpretable result even though no single")
            print("  gene passed in 10c: stiffness predicts WHICH CELLS are")
            print("  there, not how hard each cell transcribes.")

    # =========================================================================
    # D. Animal-level association, n = 8
    # =========================================================================
    print("\n" + "=" * 72)
    print("D. ANIMAL-LEVEL, n = 8")
    print("=" * 72)
    print("  10c said the association lives between animals. This quantifies it")
    print("  at that scale, where n is 8 and no amount of binning changes that.")

    a_idx = {a: np.where(animals == a)[0] for a in lev}
    a_stiff = np.array([np.median(y[a_idx[a]]) for a in lev])
    pb = np.vstack([np.asarray(agg[a_idx[a]].sum(axis=0)).ravel() for a in lev])
    depth_a = pb.sum(axis=1, keepdims=True)
    pb_cpm = np.log1p(pb / np.maximum(depth_a, 1) * SCALE_FACTOR)

    keep_g = (pb > 0).sum(axis=0) >= max(3, len(lev) // 2)
    print(f"  animals: {list(lev)}")
    print(f"  median log10 E per animal: "
          f"{dict(zip(lev, np.round(a_stiff, 3)))}")
    print(f"  genes tested: {int(keep_g.sum()):,}")

    rs, ps = [], []
    for j in np.where(keep_g)[0]:
        r, p = spearmanr(pb_cpm[:, j], a_stiff)
        rs.append(r)
        ps.append(p)
    d = pd.DataFrame({"gene": genes[keep_g], "rho_animal": rs, "p_animal": ps})
    d = d.dropna().sort_values("p_animal")
    if len(d):
        from statsmodels.stats.multitest import multipletests
        d["fdr_animal"] = multipletests(d["p_animal"], method="fdr_bh")[1]
        d.to_csv(OUT_DIR / "D_animal_level_correlations.csv", index=False)
        print(f"  genes at FDR < 0.05 with n = 8: "
              f"{int((d['fdr_animal'] < 0.05).sum()):,}")
        print("  With 8 animals the smallest possible Spearman p is ~1e-3, so")
        print("  FDR across thousands of genes is brutal. Read this as effect")
        print("  size and direction, not as a significance test.")
        print(d.head(15).to_string(index=False,
                                   float_format=lambda v: f"{v:.3f}"))

    pm.to_csv(OUT_DIR / "point_metadata.csv", index=False)
    print(f"\nAll outputs written to: {OUT_DIR}")
    print("\nDECISION TABLE")
    print("  A pass, B pass -> registration sound, the 10c null is biological")
    print("  A fail         -> the mechanical map is noise; fix upstream first")
    print("  A pass, B fail -> NI and Visium coordinate frames are misaligned")


if __name__ == "__main__":
    main()