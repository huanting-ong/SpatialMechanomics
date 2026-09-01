"""
13_merge_correlation_hits.py

Merges the correlation results across all analyses into two tables. Nothing is
recomputed; this reads the CSVs scripts 10 and 12 already wrote.

Parameters
    The two outputs read different files, and that is the point.

    correlation_hits_wide.csv reads the *_significant_only_r_values.csv
    subsets, so a blank cell means the gene did not clear the significance
    gate in that analysis. It is the source of the compartment_only counts.

    Supplementary Table 5 reads the *_volcano_table_all_genes.csv files, so a
    number is present whether or not it was significant and a blank means the
    gene was not tested. The FDR columns are what let a reader tell the two
    apart, which is why every rho carries one.

    FRAGILE_MAX_HITS    1, genes significant in only one analysis are flagged
    FULL_STATS          global, young and aged for whole tissue; global only
                        for each compartment
    SOURCE_LABELS       maps the M, G, C source codes in the gene set to
                        readable categories

Actions required
    Set IN_DIR, DEG_DIR, DEG_FILES and MECHANOMICS_CSV. All column names are
    resolved from candidate lists and printed at the start of the run, so a
    mismatch stops with the file's actual columns rather than a traceback.
        py 13_merge_correlation_hits.py

Input
    <IN_DIR>/**/*_significant_only_r_values.csv         from script 12
    <IN_DIR>/**/*_volcano_table_all_genes.csv           from script 12
    <DEG_DIR>/0_global/DEG_Global_AllCompartments_Aged_vs_Young.csv
    <DEG_DIR>/1_compartment/DEG_<compartment>_Aged_vs_Young.csv
                                                        from script 10
    mechanomics_gene_set.csv                            Supplementary Table 4
Output
    merged_correlation_hits/
        correlation_hits_wide.csv               one row per gene significant
                                                somewhere, categorised by where
        correlation_hits_long.csv               the same, one row per hit
        correlation_hits_category_summary.csv   counts by category
        Supplementary_Table_5_mechanomics_summary.csv
                                                all 1,854 Mechanomics genes
                                                with their age differential
                                                expression and stiffness
                                                correlations

Note
    |rho| is not comparable between analyses with different n. SE(rho) is
    about sqrt(1.06/(n-3)), so a smaller compartment needs a larger |rho| to
    pass the same gate, and compartment values will look stronger than
    whole-tissue values even when nothing is biologically different.
    rho_gain is reported for sorting only.

    Each analysis was FDR-corrected within itself, not across the whole set.
    A gene significant in exactly one analysis, at a rho just past threshold,
    is the single most likely thing to be a false positive. n_hits and
    fragile_single_hit are there so you can weight accordingly.
"""

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

# =============================================================================
# config
# =============================================================================
IN_DIR = Path(
    r"Z:\Visium_Data\001_SpatialX\20260803_FinalAnnotatedMetadata"
    r"\pointlevel_expression_outputs_ageAdj\Elasticity_l_Pa")

MECHANOMICS_CSV = Path(
    r"Z:\Visium_Data\001_SpatialX\20260803_FinalAnnotatedMetadata"
    r"\pointlevel_expression_outputs_ageAdj\mechanomics_gene_set.csv")

# Significant-only subsets, for output 1.
SIG_GLOB = "*_significant_only_r_values.csv"

# Full results, for output 2. One volcano table per region per statistic:
#   whole_Elasticity_l_Pa_globalcorr_volcano_table_all_genes.csv
#   CL_Elasticity_l_Pa_globalcorr_volcano_table_all_genes.csv
MECH_PARAM = "Elasticity_l_Pa"
FULL_TEMPLATE = "{region}_{param}_{stat}corr_volcano_table_all_genes.csv"

# Which statistics to pull per region. Compartments have young and aged
# volcano tables too; they are left out of Table 5 by design, because global
# per compartment is the right level for a reference table.
FULL_STATS = {
    "whole": ["global", "young", "aged"],
    "CL": ["global"],
    "S": ["global"],
    "V": ["global"],
    "F": ["global"],
}

# DESeq2 outputs from script 8. Named by biological region rather than by the
# CL/S/V/F codes used elsewhere, so the mapping is written out here instead of
# being parsed from the filename.
DEG_DIR = Path(
    r"Z:\Visium_Data\001_SpatialX\20260803_FinalAnnotatedMetadata"
    r"\deg_gsea_outputs")
DEG_FILES = {
    "whole": r"0_global\DEG_Global_AllCompartments_Aged_vs_Young.csv",
    "CL":    r"1_compartment\DEG_CL_Aged_vs_Young.csv",
    "S":     r"1_compartment\DEG_Stroma_Aged_vs_Young.csv",
    "V":     r"1_compartment\DEG_Vascular_Aged_vs_Young.csv",
    "F":     r"1_compartment\DEG_Follicle_Aged_vs_Young.csv",
}

SEARCH_RECURSIVE = True
WHOLE_LABEL = "whole"
COMPARTMENT_ORDER = ["CL", "S", "V", "F"]
STATS = ["R_global", "R_young", "R_aged"]
FRAGILE_MAX_HITS = 1

OUT_DIR = IN_DIR / "merged_correlation_hits"
OUT_WIDE = "correlation_hits_wide.csv"
OUT_LONG = "correlation_hits_long.csv"
OUT_SUMMARY = "correlation_hits_category_summary.csv"
OUT_TABLE5 = "Supplementary_Table_5_mechanomics_summary.csv"

# =============================================================================
# column name candidates, first match wins
# 'Unnamed: 0' and 'X' catch the case where R wrote gene symbols as row names.
# =============================================================================
COL_GENE = ["gene", "Gene", "gene_symbol", "symbol", "SYMBOL", "gene_name",
            "Unnamed: 0", "X"]
COL_RHO_ANY = ["rho", "R", "rho_partial", "partial_rho", "spearman_rho",
               "correlation", "R_global", "r"]
COL_FDR_ANY = ["FDR", "fdr", "padj", "p_adj", "qvalue", "q_value", "FDR_BH",
               "adj_pval", "p_adjusted"]
COL_DETECT = ["pct_detected", "frac_detected", "detection_rate", "pct_points",
              "n_points_detected", "n_points"]
COL_L2FC = ["log2FoldChange", "log2FC", "l2fc", "logFC", "avg_log2FC"]
COL_PADJ = ["padj", "p_adj", "adj_p", "FDR", "p_val_adj"]

# How a gene got onto the list. There is no connectome-only code, because a
# ligand-receptor partner only enters by partnering something already there.
SOURCE_LABELS = {
    "M":   "matrisome",
    "G":   "curated",
    "MG":  "matrisome + curated",
    "MC":  "matrisome + LR partner",
    "CG":  "curated + LR partner",
    "MCG": "matrisome + curated + LR partner",
}


# =============================================================================
# util
# =============================================================================
def pick(df, candidates, what, path, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if not required:
        return None
    print(f"\n[FATAL] no column for {what} in {path.name}")
    print(f"        tried: {candidates}")
    print(f"        file has: {list(df.columns)}")
    sys.exit(1)


def finder():
    return IN_DIR.rglob if SEARCH_RECURSIVE else IN_DIR.glob


# =============================================================================
# output 1: significant-only
# =============================================================================
def load_significant():
    it = finder()
    files = sorted(it(SIG_GLOB))
    if not files:
        raise SystemExit(f"No files matching {SIG_GLOB} under:\n  {IN_DIR}")

    frames, seen, where = [], {}, {}
    for f in files:
        reg = re.sub(r"_.*$", "", f.name)
        if reg in where:
            print(f"  [DUPLICATE] {reg}: ignoring {f.name}")
            continue
        where[reg] = f
        d = pd.read_csv(f)
        if "gene" not in d.columns:
            print(f"  [SKIP] {f.name}: no 'gene' column")
            continue
        for s in STATS:
            if s not in d.columns:
                d[s] = np.nan
        d = d[["gene"] + STATS].copy()
        d["gene"] = d["gene"].astype(str).str.strip()
        d["region"] = reg
        # An EMPTY file is a result, not a failure: that region was tested and
        # nothing passed. It must not silently vanish from the report.
        seen[reg] = {s: int(d[s].notna().sum()) for s in STATS}
        frames.append(d)
        print(f"  {reg:>6s}  {len(d):>5,} row(s)   " +
              "  ".join(f"{s}={seen[reg][s]:,}" for s in STATS))

    missing = [r for r in [WHOLE_LABEL] + COMPARTMENT_ORDER if r not in seen]
    if missing:
        print(f"\n  [WARNING] no significant-only file for: "
              f"{', '.join(missing)}")

    # Dropping empty frames before concat avoids the all-NA dtype warning.
    # `seen` still records them, so an empty region stays in the report.
    nonempty = [f for f in frames if len(f)]
    if not nonempty:
        return pd.DataFrame(columns=["gene"] + STATS + ["region"]), seen
    return pd.concat(nonempty, ignore_index=True), seen


def build(df):
    long = df.melt(id_vars=["gene", "region"], value_vars=STATS,
                   var_name="stat", value_name="rho").dropna(subset=["rho"])
    long["stat"] = long["stat"].str.replace("^R_", "", regex=True)
    long["scope"] = np.where(long["region"] == WHOLE_LABEL, "whole",
                             "compartment")
    long["analysis"] = long["region"] + ":" + long["stat"]

    wide = long.pivot_table(index="gene", columns="analysis", values="rho",
                            aggfunc="first")
    order = [f"{r}:{s}" for r in ([WHOLE_LABEL] + COMPARTMENT_ORDER)
             for s in ["global", "young", "aged"]]
    wide = wide[[c for c in order if c in wide.columns]]

    g = long.groupby("gene")
    out = pd.DataFrame(index=wide.index)
    out["n_hits"] = g.size()
    out["analyses"] = g["analysis"].apply(lambda s: ",".join(sorted(set(s))))

    whole = long[long["scope"] == "whole"]
    comp = long[long["scope"] == "compartment"]
    w_stats = whole.groupby("gene")["stat"].apply(set)
    c_regs = comp.groupby("gene")["region"].apply(lambda s: sorted(set(s)))

    out["whole_stats"] = w_stats.reindex(out.index).apply(
        lambda v: ",".join(sorted(v)) if isinstance(v, set) else "")
    out["compartments"] = c_regs.reindex(out.index).apply(
        lambda v: ",".join(v) if isinstance(v, list) else "")
    out["n_compartments"] = c_regs.reindex(out.index).apply(
        lambda v: len(v) if isinstance(v, list) else 0)

    in_whole = out["whole_stats"].str.len() > 0
    in_whole_global = out["whole_stats"].str.contains(r"\bglobal\b", regex=True)

    # The headline label: the LEAST specific analysis that found this gene.
    out["category"] = np.select(
        [in_whole_global,
         in_whole & ~in_whole_global,
         ~in_whole & (out["n_compartments"] > 0)],
        ["whole_global", "whole_age_specific", "compartment_only"],
        default="unclassified")

    out["compartment_specificity"] = np.select(
        [out["n_compartments"] == 0,
         out["n_compartments"] == 1,
         out["n_compartments"] > 1],
        ["none", "single", "multiple"], default="none")

    amax = lambda s: s.abs().max()
    out["max_abs_rho_whole"] = whole.groupby("gene")["rho"].apply(
        amax).reindex(out.index)
    out["max_abs_rho_comp"] = comp.groupby("gene")["rho"].apply(
        amax).reindex(out.index)
    out["rho_gain_comp_minus_whole"] = (out["max_abs_rho_comp"]
                                        - out["max_abs_rho_whole"])

    idx = long.loc[long.groupby("gene")["rho"].apply(
        lambda s: s.abs().idxmax())]
    out["strongest_analysis"] = idx.set_index("gene")["analysis"]
    out["strongest_rho"] = idx.set_index("gene")["rho"]

    # A gene whose sign flips between analyses is either compartment-dependent
    # biology or noise. Either way it must not be averaged.
    signs = long.assign(sg=np.sign(long["rho"])).groupby("gene")["sg"].nunique()
    out["sign_conflict"] = (signs.reindex(out.index) > 1)
    out["fragile_single_hit"] = out["n_hits"] <= FRAGILE_MAX_HITS
    out["evidence"] = long.assign(
        e=long["analysis"] + "(" + long["rho"].round(3).astype(str) + ")"
    ).groupby("gene")["e"].apply(lambda s: "; ".join(sorted(s)))

    return long, wide.join(out).reset_index()


def report(wide, seen):
    print("\n" + "=" * 70)
    print("CATEGORIES  (significant-only merge)")
    print("=" * 70)
    tab = wide["category"].value_counts()
    for k in ["whole_global", "whole_age_specific", "compartment_only"]:
        print(f"  {k:<22s} {int(tab.get(k, 0)):>6,}")
    print(f"  {'TOTAL genes':<22s} {len(wide):>6,}")

    co = wide[wide["category"] == "compartment_only"]
    print("\n  Compartment-only genes, by compartment:")
    if len(co):
        for c, n in co["compartments"].value_counts().items():
            print(f"    {c:<16s} {n:>6,}")
    else:
        print("    none")

    empty = [r for r, v in seen.items() if sum(v.values()) == 0]
    if empty:
        print(f"\n  [NOTE] zero significant genes in: {', '.join(empty)}. "
              f"That is a result, not a missing file.")

    frag = int(wide["fragile_single_hit"].sum())
    print(f"\n  Seen in only one analysis: {frag:,} of {len(wide):,} "
          f"({frag / max(len(wide), 1):.0%}). These are the most likely false "
          f"positives.")
    print(f"  Sign flips between analyses: "
          f"{int(wide['sign_conflict'].sum()):,}. Inspect before pooling.")

    g = wide["rho_gain_comp_minus_whole"].dropna()
    if len(g):
        print(f"\n  rho_gain (compartment minus whole), {len(g):,} gene(s) in "
              f"both: median {g.median():+.3f}, {int((g > 0).sum()):,} "
              f"positive.")
        print("  A positive median is EXPECTED with no biology: smaller n "
              "needs a larger |rho| to pass the same gate.")


# ============================================================================= 
# output 2: Table 5
# =============================================================================
def parse_matrisome(s):
    """Division and category out of the free-text info field.

    'M: Core matrisome | ECM Glycoproteins | ...' -> ('Core matrisome',
    'ECM Glycoproteins'). Returns (None, None) for non-matrisome genes."""
    if not isinstance(s, str):
        return pd.Series([None, None])
    m = re.search(r"M:\s*([^|]+)\|\s*([^|]+)", s)
    if not m:
        return pd.Series([None, None])
    return pd.Series([m.group(1).strip(), m.group(2).strip()])


def load_mechanomics():
    if not MECHANOMICS_CSV.exists():
        print(f"\n[FATAL] gene set not found:\n  {MECHANOMICS_CSV}")
        sys.exit(1)
    d = pd.read_csv(MECHANOMICS_CSV)
    for c in ("gene", "source", "info", "in_expression_matrix"):
        if c not in d.columns:
            print(f"\n[FATAL] '{c}' missing from {MECHANOMICS_CSV.name}")
            print(f"        file has: {list(d.columns)}")
            sys.exit(1)

    out = pd.DataFrame({"gene": d["gene"].astype(str).str.strip()})
    out["category"] = d["source"].map(SOURCE_LABELS).fillna(d["source"])
    out[["matrisome_division", "matrisome_category"]] = \
        d["info"].apply(parse_matrisome)
    out["in_matrisome"] = d.get("in_matrisome")
    out["in_curated"] = d.get("in_goi")
    out["connectome_role"] = d.get("connectome_role")
    out["in_expression_matrix"] = d["in_expression_matrix"]
    out["provenance"] = d["info"]

    out = out.drop_duplicates("gene").reset_index(drop=True)
    print(f"  {len(out):,} symbols, "
          f"{int(out['in_expression_matrix'].sum()):,} in the matrix")
    print(f"  category  : {out['category'].value_counts().to_dict()}")
    print(f"  matrisome : "
          f"{out['matrisome_category'].value_counts(dropna=False).to_dict()}")
    return out


def load_full_correlations():
    """One volcano table per region per statistic.

    Column names are resolved from candidates and printed, because these files
    were written by a different part of script 10 than the merged r_values
    files and do not share their naming."""
    it = finder()
    cols = {}
    for reg, stats in FULL_STATS.items():
        keep = None
        for s in stats:
            name = FULL_TEMPLATE.format(region=reg, param=MECH_PARAM, stat=s)
            hits = [p for p in it(name) if not p.name.startswith("._")]
            if not hits:
                print(f"  [MISSING] {reg}:{s}   {name}")
                continue
            f = hits[0]
            d = pd.read_csv(f)
            gcol = pick(d, COL_GENE, "gene symbol", f)
            rc = pick(d, COL_RHO_ANY, f"rho ({reg}:{s})", f)
            fc = pick(d, COL_FDR_ANY, f"FDR ({reg}:{s})", f, required=False)
            part = pd.DataFrame({
                "gene": d[gcol].astype(str).str.strip(),
                f"{reg}_rho_{s}": pd.to_numeric(d[rc], errors="coerce"),
            })
            if fc:
                part[f"{reg}_rho_{s}_FDR"] = pd.to_numeric(d[fc],
                                                           errors="coerce")
            if reg == WHOLE_LABEL and s == "global":
                dc = pick(d, COL_DETECT, "detection rate", f, required=False)
                if dc:
                    part["pct_measurements_detected"] = pd.to_numeric(
                        d[dc], errors="coerce")
            part = part.drop_duplicates("gene")
            keep = part if keep is None else keep.merge(part, on="gene",
                                                        how="outer")
            print(f"  {reg:>6s}:{s:<7s} {len(part):>6,} genes   "
                  f"rho={rc}   FDR={fc or 'NOT PRESENT'}")
        if keep is not None:
            cols[reg] = keep
    if not cols:
        print(f"\n[FATAL] no volcano tables found under {IN_DIR}")
        print(f"        expected e.g. "
              f"{FULL_TEMPLATE.format(region='whole', param=MECH_PARAM, stat='global')}")
        sys.exit(1)
    return cols


def load_deg():
    cols = {}
    for reg, rel in DEG_FILES.items():
        f = DEG_DIR / rel
        if not f.exists():
            print(f"  [MISSING] {reg}: {f}")
            continue
        d = pd.read_csv(f)
        gcol = pick(d, COL_GENE, "gene symbol", f)
        lc = pick(d, COL_L2FC, "log2 fold change", f)
        pc = pick(d, COL_PADJ, "adjusted p", f)
        cols[reg] = pd.DataFrame({
            "gene": d[gcol].astype(str).str.strip(),
            f"{reg}_YvA_L2FC": pd.to_numeric(d[lc], errors="coerce"),
            f"{reg}_YvA_padj": pd.to_numeric(d[pc], errors="coerce"),
        }).drop_duplicates("gene")
        print(f"  {reg:>6s}  {len(cols[reg]):>6,} genes   "
              f"L2FC={lc}  padj={pc}   <- {f.name}")
    if not cols:
        print(f"\n[FATAL] no DEG files found under {DEG_DIR}")
        sys.exit(1)
    return cols


def build_table5(mech, corr, deg, sig_wide):
    t = mech.copy()

    for reg in [WHOLE_LABEL] + COMPARTMENT_ORDER:
        if reg in deg:
            t = t.merge(deg[reg], on="gene", how="left")
    for reg in [WHOLE_LABEL] + COMPARTMENT_ORDER:
        if reg in corr:
            t = t.merge(corr[reg], on="gene", how="left")

    # in_expression_matrix comes from the gene set file. tested_correlation is
    # the second flag a reader needs: a gene can be in the matrix and still
    # fall below the detection filter for correlation in a given region.
    rho_cols = [c for c in t.columns
                if re.search(r"_rho_(global|young|aged)$", c)]
    t["tested_correlation"] = (t[rho_cols].notna().any(axis=1)
                               if rho_cols else False)

    sc = sig_wide.set_index("gene")["sign_conflict"] if len(sig_wide) else None
    t["sign_conflict"] = (t["gene"].map(sc).fillna(False)
                          if sc is not None else False)

    order = ["gene", "category", "in_matrisome", "in_curated",
             "connectome_role", "matrisome_division", "matrisome_category",
             "in_expression_matrix", "tested_correlation",
             "pct_measurements_detected"]
    order += [f"{r}_YvA_{s}" for r in [WHOLE_LABEL] + COMPARTMENT_ORDER
              for s in ["L2FC", "padj"]]
    order += [f"{WHOLE_LABEL}_rho_{s}{sfx}"
              for s in ["global", "young", "aged"] for sfx in ["", "_FDR"]]
    order += [f"{r}_rho_global{sfx}" for r in COMPARTMENT_ORDER
              for sfx in ["", "_FDR"]]
    order += ["sign_conflict", "provenance"]

    t = t[[c for c in order if c in t.columns]]
    return t.sort_values(["category", "gene"]).reset_index(drop=True)


def report_table5(t):
    print("\n" + "=" * 70)
    print("SUPPLEMENTARY TABLE 5")
    print("=" * 70)
    print(f"  {len(t):,} genes x {len(t.columns)} columns")
    print(f"  in expression matrix   : "
          f"{int(t['in_expression_matrix'].sum()):,}")
    print(f"  tested for correlation : "
          f"{int(t['tested_correlation'].sum()):,}")
    print("\n  non-empty values per data column:")
    skip = {"gene", "category", "in_matrisome", "in_curated",
            "connectome_role", "matrisome_division", "matrisome_category",
            "provenance"}
    for c in t.columns:
        if c in skip:
            continue
        n = int(t[c].notna().sum())
        flag = "   <- EMPTY, check the join" if n == 0 else ""
        print(f"    {c:<34s} {n:>6,}{flag}")


def main():
    print("=" * 70)
    print("MERGE CORRELATION HITS")
    print(f"Input : {IN_DIR}")
    print("=" * 70)

    print("\n[1/4] significant-only files")
    df, seen = load_significant()
    long, wide = build(df)

    print("\n[2/4] Mechanomics gene set")
    mech = load_mechanomics()

    print("\n[3/4] full correlation results (volcano tables)")
    corr = load_full_correlations()

    print("\n[4/4] differential expression results")
    deg = load_deg()

    table5 = build_table5(mech, corr, deg, wide)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT_DIR / OUT_WIDE, index=False)
    long.to_csv(OUT_DIR / OUT_LONG, index=False)
    (wide.groupby(["category", "compartment_specificity"]).size()
        .rename("n_genes").reset_index()
        .to_csv(OUT_DIR / OUT_SUMMARY, index=False))
    table5.to_csv(OUT_DIR / OUT_TABLE5, index=False)

    report(wide, seen)
    report_table5(table5)
    print(f"\nWrote:\n  {OUT_DIR / OUT_WIDE}\n  {OUT_DIR / OUT_LONG}"
          f"\n  {OUT_DIR / OUT_SUMMARY}\n  {OUT_DIR / OUT_TABLE5}")


if __name__ == "__main__":
    main()