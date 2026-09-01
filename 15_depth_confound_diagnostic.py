"""
15_depth_confound_diagnostic.py

Standalone diagnostic showing why gene-stiffness correlation cannot be
computed at 4 um bin level, and why script 12 aggregates to the 50 um
measurement first. Reads script 12's inputs, writes nothing into its output
folder, and re-runs nothing.

The question
    A bin-level volcano is one-sided: essentially every gene correlates
    positively with stiffness and almost none negatively. In real tissue some
    genes fall where stiffness rises, so a one-sided distribution points at a
    shared technical driver rather than at biology.

The hypothesis
    Expression is log-normalised as x_g = log1p(c_g / N * SF), where c_g is the
    raw count for gene g in that bin, N is the bin's total count and SF is a
    scale factor. At 4 um most detected genes have c_g of 1 or 2, so once the
    analysis conditions on detection, the remaining variation in x_g is
    dominated by 1/N, which is shared by every gene. If that is what is
    happening then rho(x_g, E) is approximately -rho(N, E) for almost all
    genes, and one number explains the entire volcano.

What it computes
    A   rho(nCount, E) and rho(nFeature, E) per bin, globally and per
        compartment. The single number the hypothesis rests on
    B   Per-gene rho three ways on the same gene set: raw with zeros included,
        conditioned on detection, and conditioned with nFeature partialled out
        of both variables. If the skew is a depth artefact, the partial
        version becomes roughly symmetric
    C   The same per-gene rho after collapsing bins to nano_point_id. About 92
        bins share one measurement, so bin-level n is not the number of
        independent observations, which is why bin-level FDR values reach
        1e-300
    D   Reconstructs the bin-level mean expression metric and correlates it
        with nFeature, to show what that quantity actually measures

Actions required
    Set METADATA_TSV. Everything else is read from the same expression_export
    folder script 12 uses.
        py 15_depth_confound_diagnostic.py

Input
    metadata_with_nano.tsv                  from script 9
    expression_export/                      from script 10
Output
    Diagnostic tables and figures in their own folder, including the
    three-way rho distribution and the nFeature against stiffness plot.
    Evidence for Supplementary Methods S9.

Note
    nCount_RNA cannot be recovered from a log-normalised matrix, so part A
    needs the nCount_RNA and nFeature_RNA columns carried through the script
    10 export. If they are absent, nFeature is recomputed from the matrix,
    which is exact, and nCount is reported as unavailable.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io as sio
from scipy import sparse
from scipy.stats import spearmanr, rankdata
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG  (mirrors script 12 so the comparison is like for like)
# =============================================================================

METADATA_TSV = Path(
    r"Z:\Visium_Data\001_SpatialX\20260803_FinalAnnotatedMetadata\metadata_with_nano.tsv"
)
EXPR_DIR = METADATA_TSV.parent / "expression_export"
OUT_DIR = METADATA_TSV.parent / "depth_confound_diagnostic"

TARGET_COL = "Elasticity_l_Pa"
LOG_TARGET = True
EXPRESSION_THRESH = 0.03      # same as script 12
QC_THRESH = 0.05              # same as script 12
NANO_ID_COL = "nano_point_id"
COMPARTMENT_COL = "Compartments"

N_GENES_TEST = 1500           # genes sampled for the per-gene comparison
MIN_BINS_PER_POINT = 5        # same as script 12
RANDOM_STATE = 42

EXPR_MATRIX_FILE = "step9_expression_matrix.mtx"
EXPR_BARCODES_FILE = "step9_expression_barcodes.txt"
EXPR_FEATURES_FILE = "step9_expression_features.txt"
EXPR_MATCHED_METADATA_FILE = "step9_expression_metadata_matched.tsv"

# =============================================================================
# HELPERS
# =============================================================================


def partial_spearman(x, y, z):
    """
    Spearman partial correlation of x and y controlling for z.

    Rank-transform all three, then apply the first-order partial correlation
    formula to the ranks. This is the standard definition of partial Spearman.
    Returns nan when a variable has no rank variation.
    """
    if len(x) < 10:
        return np.nan
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    if np.std(rx) == 0 or np.std(ry) == 0 or np.std(rz) == 0:
        return np.nan
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return (rxy - rxz * ryz) / denom


def load_expression(expr_dir):
    print(f"Loading expression from {expr_dir}")
    mat = sio.mmread(expr_dir / EXPR_MATRIX_FILE).tocsr()
    with open(expr_dir / EXPR_BARCODES_FILE, encoding="utf-8") as f:
        barcodes = [l.strip() for l in f]
    with open(expr_dir / EXPR_FEATURES_FILE, encoding="utf-8") as f:
        genes = [l.strip() for l in f]

    # R writes genes x cells; script 12 transposes on the same condition
    if mat.shape[0] == len(genes) and mat.shape[1] == len(barcodes):
        mat = mat.T.tocsr()

    if mat.shape[0] != len(barcodes) or mat.shape[1] != len(genes):
        raise ValueError(
            f"Shape mismatch: matrix {mat.shape}, "
            f"{len(barcodes)} barcodes, {len(genes)} genes"
        )
    print(f"  {mat.shape[0]:,} bins x {mat.shape[1]:,} genes")
    return mat, np.array(barcodes), np.array(genes)


def clean_bool(series):
    return series.astype(str).str.strip().str.lower() == "true"


def summarise(name, rho, p, n):
    print(f"  {name:<34s} rho = {rho:+.4f}   p = {p:.3g}   n = {n:,}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    X, barcodes, genes = load_expression(EXPR_DIR)
    meta = pd.read_csv(EXPR_DIR / EXPR_MATCHED_METADATA_FILE,
                       sep="\t", low_memory=False)
    print(f"  matched metadata rows: {len(meta):,}")

    if len(meta) != X.shape[0]:
        raise ValueError(
            f"Metadata rows ({len(meta)}) != matrix rows ({X.shape[0]}). "
            "The export and the metadata are out of sync; re-run the script 9 "
            "export chunk before trusting anything below."
        )

    # -------------------------------------------------------------------------
    # Row filter, matching script 12
    # -------------------------------------------------------------------------
    meta[TARGET_COL] = pd.to_numeric(meta[TARGET_COL], errors="coerce")
    keep = meta[TARGET_COL].notna()
    if "valid_measurement" in meta.columns:
        keep &= clean_bool(meta["valid_measurement"])
    if NANO_ID_COL in meta.columns:
        keep &= meta[NANO_ID_COL].notna()
        keep &= meta[NANO_ID_COL].astype(str).str.strip().ne("N/A")
    keep = keep.values

    X = X[keep]
    meta = meta.loc[keep].reset_index(drop=True)
    print(f"  bins after filtering: {X.shape[0]:,}")

    y = meta[TARGET_COL].values.astype(float)
    if LOG_TARGET and TARGET_COL in {"Elasticity_l_Pa", "Elasticity_eff_Pa"}:
        y = np.log10(y)
        print(f"  {TARGET_COL} log10-transformed, as in script 10")

    # -------------------------------------------------------------------------
    # Depth metrics
    # nFeature is recomputed from the matrix and is exact: the data layer is
    # log1p of a non-negative quantity, so x > 0 is exactly 'gene detected'.
    # nCount cannot be recovered from normalised data.
    # -------------------------------------------------------------------------
    n_feature = np.asarray((X > 0).sum(axis=1)).ravel().astype(float)
    have_ncount = "nCount_RNA" in meta.columns
    n_count = meta["nCount_RNA"].values.astype(float) if have_ncount else None

    if have_ncount:
        agree = spearmanr(n_count, n_feature)
        print(f"  nCount_RNA present. rho(nCount, nFeature) = {agree.correlation:+.3f}")
    else:
        print("  nCount_RNA NOT in the export. Re-run the script 9 export chunk "
              "with the nCount_RNA / nFeature_RNA lines to enable that test. "
              "Continuing with nFeature only.")

    # =========================================================================
    # A. Does sequencing depth track stiffness?
    # =========================================================================
    print("\n" + "=" * 72)
    print("A. DEPTH vs STIFFNESS  (the single number the hypothesis rests on)")
    print("=" * 72)

    rows = []

    def depth_block(label, mask):
        n = int(mask.sum())
        if n < 50:
            print(f"  {label}: only {n} bins, skipped")
            return
        print(f"\n {label}  (n = {n:,} bins)")
        r_f, p_f = spearmanr(n_feature[mask], y[mask])
        summarise("rho(nFeature, log10 E)", r_f, p_f, n)
        rows.append(dict(group=label, metric="nFeature", rho=r_f, pval=p_f, n=n))
        if have_ncount:
            r_c, p_c = spearmanr(n_count[mask], y[mask])
            summarise("rho(nCount, log10 E)", r_c, p_c, n)
            rows.append(dict(group=label, metric="nCount", rho=r_c, pval=p_c, n=n))
        print(f"   -> predicted per-gene rho if depth-driven: "
              f"{-r_f:+.3f} (sign flips, because x ~ 1/N)")

    all_mask = np.ones(len(y), dtype=bool)
    depth_block("global", all_mask)

    if COMPARTMENT_COL in meta.columns:
        for comp, sub in meta.groupby(COMPARTMENT_COL).groups.items():
            m = np.zeros(len(y), dtype=bool)
            m[np.asarray(sub)] = True
            depth_block(f"compartment {comp}", m)

    pd.DataFrame(rows).to_csv(OUT_DIR / "A_depth_vs_stiffness.csv", index=False)

    # =========================================================================
    # B. Per-gene rho: raw vs conditioned vs partial
    # =========================================================================
    print("\n" + "=" * 72)
    print("B. PER-GENE rho, THREE WAYS")
    print("=" * 72)

    detect_rate = np.asarray((X > EXPRESSION_THRESH).sum(axis=0)).ravel() / X.shape[0]
    eligible = np.where(detect_rate > QC_THRESH)[0]
    print(f"  genes passing detection > {QC_THRESH:.0%}: {len(eligible):,}")
    if len(eligible) == 0:
        raise SystemExit("No genes pass the detection filter; nothing to test.")

    sel = (eligible if len(eligible) <= N_GENES_TEST
           else rng.choice(eligible, N_GENES_TEST, replace=False))
    sel = np.sort(sel)
    print(f"  testing a random sample of {len(sel):,} of them "
          f"(seed {RANDOM_STATE}, so this is reproducible)")

    Xc = X[:, sel].tocsc()
    recs = []
    for j, gi in enumerate(sel):
        xg = np.asarray(Xc[:, j].todense()).ravel()

        r_raw, p_raw = spearmanr(xg, y)

        m = xg > EXPRESSION_THRESH
        if m.sum() < 20:
            continue
        r_con, p_con = spearmanr(xg[m], y[m])
        r_par = partial_spearman(xg[m], y[m], n_feature[m])

        recs.append(dict(gene=genes[gi],
                         detect_rate=detect_rate[gi],
                         n_conditioned=int(m.sum()),
                         rho_raw=r_raw, p_raw=p_raw,
                         rho_conditioned=r_con, p_conditioned=p_con,
                         rho_partial_nFeature=r_par))
        if (j + 1) % 250 == 0:
            print(f"    {j + 1:,}/{len(sel):,} genes")

    gdf = pd.DataFrame(recs).dropna(subset=["rho_conditioned"])
    gdf["fdr_conditioned"] = multipletests(gdf["p_conditioned"], method="fdr_bh")[1]
    gdf.to_csv(OUT_DIR / "B_per_gene_rho_three_ways.csv", index=False)

    def frac_pos(col):
        v = gdf[col].dropna()
        return (v > 0).mean(), v.median()

    print("\n  fraction of genes with rho > 0, and median rho:")
    for col in ["rho_raw", "rho_conditioned", "rho_partial_nFeature"]:
        f, m_ = frac_pos(col)
        print(f"    {col:<24s} {f:6.1%} positive   median {m_:+.3f}")
    print("\n  A one-sided volcano means 'conditioned' is near 100% positive.")
    print("  If 'partial' drops toward 50%, the skew was sequencing depth.")
    print("  If 'partial' stays one-sided, depth is NOT the explanation and the")
    print("  cause is elsewhere. Do not assume; read the numbers.")

    # =========================================================================
    # C. Pseudoreplication: collapse bins to nanoindentation points
    # =========================================================================
    print("\n" + "=" * 72)
    print("C. SAME GENES AT NANOINDENTATION-POINT LEVEL")
    print("=" * 72)

    if NANO_ID_COL not in meta.columns:
        print("  no nano_point_id column, skipped")
    else:
        pid = meta[NANO_ID_COL].astype(str).values
        uniq, inv = np.unique(pid, return_inverse=True)
        counts = np.bincount(inv).astype(float)
        ind = sparse.csr_matrix(
            (np.ones(len(inv)), (inv, np.arange(len(inv)))),
            shape=(len(uniq), len(inv)))
        agg = ind @ Xc
        agg = sparse.csr_matrix(agg).multiply(1.0 / counts[:, None]).tocsc()

        y_pt = np.bincount(inv, weights=y) / counts
        ok = counts >= MIN_BINS_PER_POINT
        print(f"  bins: {X.shape[0]:,}  ->  nano points: {ok.sum():,} "
              f"(>= {MIN_BINS_PER_POINT} bins each)")
        print(f"  median bins per point: {np.median(counts):.0f}")
        print("  Bin-level n is NOT the number of independent observations.")
        print("  That is why bin-level FDR reaches 1e-300 and the y axis clips.")

        precs = []
        for j in range(agg.shape[1]):
            v = np.asarray(agg[:, j].todense()).ravel()[ok]
            if np.std(v) == 0:
                continue
            r, p = spearmanr(v, y_pt[ok])
            precs.append(dict(gene=genes[sel[j]], rho_point=r, p_point=p))
        pdf = pd.DataFrame(precs).dropna(subset=["rho_point"])
        if len(pdf):
            pdf["fdr_point"] = multipletests(pdf["p_point"], method="fdr_bh")[1]
            pdf.to_csv(OUT_DIR / "C_per_gene_rho_point_level.csv", index=False)
            print(f"\n  point-level: {(pdf['rho_point'] > 0).mean():6.1%} positive, "
                  f"median rho {pdf['rho_point'].median():+.3f}")
            print(f"  genes at FDR < 0.01 point-level : "
                  f"{(pdf['fdr_point'] < 0.01).sum():,} / {len(pdf):,}")
            print(f"  genes at FDR < 0.01 bin-level   : "
                  f"{(gdf['fdr_conditioned'] < 0.01).sum():,} / {len(gdf):,}")

            merged = gdf.merge(pdf, on="gene", how="inner")
            merged.to_csv(OUT_DIR / "C_bin_vs_point_merged.csv", index=False)

    # =========================================================================
    # D. What script 12's mean_gene_expression actually measures
    # =========================================================================
    print("\n" + "=" * 72)
    print("D. mean_gene_expression vs nFeature")
    print("=" * 72)

    Xe = X[:, eligible].tocsr()
    det = Xe > EXPRESSION_THRESH
    n_det = np.asarray(det.sum(axis=1)).ravel().astype(float)
    sum_det = np.asarray(Xe.multiply(det).sum(axis=1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_expr = np.where(n_det > 0, sum_det / n_det, np.nan)

    ok2 = np.isfinite(mean_expr)
    r1, p1 = spearmanr(mean_expr[ok2], n_feature[ok2])
    r2, p2 = spearmanr(mean_expr[ok2], y[ok2])
    summarise("rho(mean_gene_expr, nFeature)", r1, p1, int(ok2.sum()))
    summarise("rho(mean_gene_expr, log10 E)", r2, p2, int(ok2.sum()))
    print("\n  Library-size normalisation fixes the un-logged total per bin, so")
    print("  the mean over DETECTED genes is close to scale_factor / nFeature.")
    print("  A strongly negative rho(mean_gene_expr, nFeature) confirms that the")
    print("  mean_gene_correlation figure is largely a complexity plot.")

    pd.DataFrame({"mean_gene_expression": mean_expr,
                  "nFeature": n_feature,
                  "log10_target": y,
                  COMPARTMENT_COL: meta[COMPARTMENT_COL]
                  if COMPARTMENT_COL in meta.columns else np.nan}
                 ).to_csv(OUT_DIR / "D_mean_expression_per_bin.csv", index=False)

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, col, ttl in zip(
        axes,
        ["rho_raw", "rho_conditioned", "rho_partial_nFeature"],
        ["raw (all bins)", "conditioned (x > 0.03)", "partial, nFeature removed"],
    ):
        v = gdf[col].dropna()
        ax.hist(v, bins=60, color="#4C72B0")
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_title(f"{ttl}\n{(v > 0).mean():.0%} positive", fontsize=9)
        ax.set_xlabel("Spearman rho vs log10 E")
        ax.set_ylabel("genes")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"rho_distribution_three_ways.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)

    step = max(1, len(y) // 40000)   # thin for file size only
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.scatter(n_feature[::step], y[::step], s=3, alpha=0.25,
               color="grey", rasterized=True, linewidths=0)
    r_f, p_f = spearmanr(n_feature, y)
    ax.set_xlabel("Genes detected per bin (nFeature)")
    ax.set_ylabel(f"log10 {TARGET_COL}")
    ax.set_title(f"rho = {r_f:+.3f}, p = {p_f:.2g}, n = {len(y):,}", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"nFeature_vs_{TARGET_COL}.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll diagnostic outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()