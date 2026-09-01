"""
5_NI_stiffness_stats.py

Stiffness statistics that respect the nesting in the design. Indentations sit
inside a region of interest, inside an ovary, inside an animal. Treating them
as independent tests the age difference against within-tissue variation
instead of between-animal variation, and returns a p far too small.

Fits
    log10(stiffness) ~ age                  is stiffness different with age?
    log10(stiffness) ~ age * compartment    do compartments differ within an
                                            age group, and does the same
                                            compartment differ between them?
    Random intercepts for animal, ovary within animal, and ROI within ovary.
    Log-transformed because the distribution is log-normal, which is also why
    Fig. 2b uses a log axis.
    Compartment contrasts are Holm-corrected within three families: pairwise
    within young, pairwise within aged, and same-compartment between ages.

Parameters
    MODE                  "points" reads the merged per-bin table, which
                          carries compartments and answers both questions but
                          covers only indentations inside a transcriptomic ROI.
                          "curves" reads the all-samples CSVs, which covers
                          every ovary indented but has no compartments
    COMPARTMENT_MIN_FRAC  0.50, the fraction of 4 um bins beneath a
                          measurement that must share a compartment for that
                          measurement to be assigned to it
    VALID_MIN, VALID_MAX  7.6 to 22,726 Pa
    PERR_MAX              0.15, applied in "curves" mode only

Actions required
    Fill SAMPLE_SHEET before reading any p value. Two ovaries from one mouse
    must share an animal id, or the model treats them as independent and the
    age test becomes too generous.
        py 5_NI_stiffness_stats.py

Input
    metadata_with_nano.tsv                          from script 9, MODE="points"
    <root>/<slice>/output/*all_samples.csv          from script 1, MODE="curves"
Output
    stiffness_stats/stiffness_observations.csv              one row per
                                                            measurement
    stiffness_stats/compartment_contrasts.csv               all 16 contrasts
                                                            with Holm-corrected p
    stiffness_stats/prism_stiffness_by_ovary_compartment_Pa.csv
    stiffness_stats/prism_stiffness_by_ovary_compartment_kPa.csv
    stiffness_stats/prism_NESTED_stiffness_kPa.csv          8 groups x 4 ovary
                                                            subcolumns for a
                                                            Prism nested table

Note
    Q1 is reported three ways: the mixed model, a t test on per-ovary means,
    and a Mann-Whitney on per-ovary means. With few animals the mixed model's
    variance components are unstable, so if it disagrees with the other two,
    the per-ovary tests are the ones to report. The script prints the variance
    components; a component at zero means that level of nesting contributed
    nothing.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

import statsmodels.formula.api as smf
from patsy import dmatrix

# =============================================================================
# CONFIG
# =============================================================================

# ---- WHERE THE DATA COMES FROM ----------------------------------------------
# Mode "points"  : the per-bin table with nanoindentation values merged in.
#                  Has compartments, so it can answer Q1 and Q2, but only
#                  covers indentations inside a transcriptomic ROI.
# Mode "curves"  : the all-samples CSVs under each slice folder.
#                  No compartments, so Q1 only, but covers every ovary you
#                  have ever indented, which is what Q1 should use.
MODE = "points"

POINTS_TSV = Path(
    r"Z:\Visium_Data\001_SpatialX\20260803_FinalAnnotatedMetadata"
    r"\metadata_with_nano.tsv")
CURVES_ROOT = Path(r"Z:\Nanoindentation\Ting-Nanoindentation2025")

OUT_DIR = Path(__file__).resolve().parent / "stiffness_stats"

# ---- THE SAMPLE SHEET -------------------------------------------------------
# key   = the slice or ROI label as it appears in the data
# ovary = the ovary that label came from
# animal= the mouse that ovary came from. Two ovaries from one mouse share it.
# age   = "young" or "aged"
#
# Two ovaries from the same animal MUST share an animal id, or the model will
# treat them as independent and the age test will be too generous.
SAMPLE_SHEET = [
    # key,     ovary, animal, age
    ("Y1-1",   "Y1",  "mY1",  "young"),
    ("Y1-2",   "Y1",  "mY1",  "young"),
    ("Y1-3",   "Y1",  "mY1",  "young"),
    ("Y1-4",   "Y1",  "mY1",  "young"),
    ("Y2-1",   "Y2",  "mY1",  "young"),   # Y1 and Y2 are one animal
    ("Y2-2",   "Y2",  "mY1",  "young"),
    ("Y4-1",   "Y4",  "mY4",  "young"),
    ("Y4-2",   "Y4",  "mY4",  "young"),
    ("Y5-1",   "Y5",  "mY5",  "young"),
    ("Y5-2",   "Y5",  "mY5",  "young"),
    ("Y5-3",   "Y5",  "mY5",  "young"),
    ("A1-1",   "A1",  "mA1",  "aged"),
    ("A1-2",   "A1",  "mA1",  "aged"),
    ("A2-1",   "A2",  "mA1",  "aged"),    # A1 and A2 are one animal
    ("A2-2",   "A2",  "mA1",  "aged"),
    ("A2-3",   "A2",  "mA1",  "aged"),
    ("A3-1",   "A3",  "mA3",  "aged"),
    ("A3-2",   "A3",  "mA3",  "aged"),
    ("A3-3",   "A3",  "mA3",  "aged"),
    ("A6-1",   "A6",  "mA6",  "aged"),
    ("A6-2",   "A6",  "mA6",  "aged"),
]

# ---- COLUMN NAMES IN THE POINTS TABLE ---------------------------------------
# Each is a list of candidates. The first one present in the file is used, and
# if none is found the script prints every column in the file and stops, so a
# missing name never turns into a traceback three functions later.
COL_STIFFNESS_CANDIDATES = ["Elasticity_l_Pa", "Elasticity_l (Pa)",
                            "Elasticity_l", "Elasticity_l_pa"]
COL_ROI_CANDIDATES = ["ROIs", "ROI", "roi", "ROI-Segmentation"]
COL_POINT_CANDIDATES = ["nano_point_id", "nano_point", "point_id",
                        "NanoPointID"]
# Optional. If absent, the composite key is ROI plus point id, which is already
# unique when ROI labels carry the ovary (Y1-1, A2-3 and so on).
COL_SLICE_CANDIDATES = ["Tissue.slices", "Tissue_slices", "Tissue slices",
                        "tissue_slice", "slice", "Slice", "samples", "Sample"]
# Candidates for the compartment column. First one present is used.
COL_COMPARTMENT_CANDIDATES = ["compartment", "Compartment", "compartments",
                              "Compartments", "compartment_4", "Region"]

COMPARTMENT_ORDER = ["CL", "S", "V", "F"]
COMPARTMENT_MIN_FRAC = 0.50   # matches the Methods

VALID_MIN, VALID_MAX = 7.6, 22726.0
PERR_MAX = 0.15
ENCODINGS = ["utf-8-sig", "cp1252", "latin-1"]


# =============================================================================
def sheet_frame():
    s = pd.DataFrame(SAMPLE_SHEET, columns=["key", "ovary", "animal", "age"])
    if s["key"].duplicated().any():
        sys.exit("SAMPLE_SHEET has duplicate keys.")
    # An ovary must belong to exactly one animal and one age.
    for col in ("animal", "age"):
        bad = s.groupby("ovary")[col].nunique()
        if (bad > 1).any():
            sys.exit(f"SAMPLE_SHEET: ovary maps to >1 {col}: "
                     f"{list(bad[bad > 1].index)}")
    return s


def read_any(path, **kw):
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"could not decode {path}")


def pick_col(df, candidates, what, required=True):
    """First candidate present, or a readable failure listing what is there."""
    for c in candidates:
        if c in df.columns:
            return c
    if not required:
        return None
    print(f"\n[FATAL] no column found for {what}.")
    print(f"        tried: {candidates}")
    print(f"        file has {len(df.columns)} columns:")
    for c in df.columns:
        print(f"          {c}")
    sys.exit(1)


def load_points(sheet):
    """Per-measurement table with compartment, from the merged metadata."""
    d = read_any(POINTS_TSV, sep="\t", low_memory=False)

    col_e = pick_col(d, COL_STIFFNESS_CANDIDATES, "stiffness")
    col_roi = pick_col(d, COL_ROI_CANDIDATES, "ROI")
    col_pt = pick_col(d, COL_POINT_CANDIDATES, "nanoindentation point id")
    col_sl = pick_col(d, COL_SLICE_CANDIDATES, "tissue slice", required=False)
    comp_col = pick_col(d, COL_COMPARTMENT_CANDIDATES, "compartment",
                        required=False)

    print(f"  stiffness   -> {col_e}")
    print(f"  ROI         -> {col_roi}")
    print(f"  point id    -> {col_pt}")
    print(f"  slice       -> {col_sl or 'not present, using ROI + point id'}")
    print(f"  compartment -> {comp_col or 'NOT FOUND, Q2 will be skipped'}")
    if comp_col is None:
        print("    columns containing comp/region/cluster/anno:")
        for c in d.columns:
            if any(k in c.lower() for k in
                   ("comp", "region", "cluster", "anno", "cell")):
                print(f"      {c}")

    d[col_e] = pd.to_numeric(d[col_e], errors="coerce")
    d = d[d[col_e].between(VALID_MIN, VALID_MAX)].copy()

    pid = d[col_pt].astype(str).str.strip()
    d = d[~pid.isin(["N/A", "nan", "", "None"])].copy()

    parts = [d[col_roi].astype(str).str.strip(),
             d[col_pt].astype(str).str.strip()]
    if col_sl is not None:
        parts.insert(0, d[col_sl].astype(str).str.strip())
    d["_point"] = parts[0]
    for p in parts[1:]:
        d["_point"] = d["_point"] + "|" + p

    rows = []
    for pt, g in d.groupby("_point"):
        roi = str(g[col_roi].iloc[0]).strip()
        e = g[col_e].dropna()
        if e.empty or e.nunique() > 1:
            continue                      # key check: one modulus per point
        rec = dict(point=pt, key=roi, stiffness=float(e.iloc[0]),
                   n_bins=len(g), compartment=np.nan)
        if comp_col is not None:
            frac = g[comp_col].value_counts(normalize=True)
            if len(frac) and frac.iloc[0] >= COMPARTMENT_MIN_FRAC:
                rec["compartment"] = str(frac.index[0]).strip()
        rows.append(rec)
    pts = pd.DataFrame(rows)
    if pts.empty:
        sys.exit("No measurements survived the key check.")
    pts = pts[pts["n_bins"] >= 5]

    merged = pts.merge(sheet, on="key", how="inner")
    lost = sorted(set(pts["key"]) - set(sheet["key"]))
    if lost:
        print(f"\n  [NOTE] {len(lost)} ROI label(s) in the data are absent "
              f"from SAMPLE_SHEET and were dropped:")
        print(f"         {lost}")
    return merged


def load_curves(sheet):
    """Every indentation from every labelled slice folder. No compartments."""
    frames = []
    for f in sorted(p for p in CURVES_ROOT.iterdir() if p.is_dir()):
        out = f / "output"
        if not out.is_dir():
            continue
        cands = [p for p in sorted(out.glob("*.csv"))
                 if not p.name.startswith("._")
                 and "all" in p.name.lower() and "samples" in p.name.lower()]
        pref = out / f"NI_{f.name}_all_samples.csv"
        path = pref if pref in cands else next(
            (p for p in cands if p.name.lower() == "all samples.csv"), None)
        if path is None:
            continue
        hit = [k for k in sheet["key"] if f.name.upper().endswith(k.upper())]
        if not hit:
            continue
        d = read_any(path)
        d.columns = [c.strip() for c in d.columns]
        ren = {c: c.split("(")[0].strip().replace(" ", "_") for c in d.columns}
        d = d.rename(columns=ren)
        d = d.loc[:, ~pd.Index(d.columns).duplicated(keep="first")]
        if "Elasticity_l" not in d.columns:
            continue
        for c in ("Elasticity_l", "Perr_l"):
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        m = d["Elasticity_l"].between(VALID_MIN, VALID_MAX)
        if "Perr_l" in d.columns:
            m &= d["Perr_l"] <= PERR_MAX
        d = d[m]
        frames.append(pd.DataFrame(dict(stiffness=d["Elasticity_l"].values,
                                        key=hit[0], compartment=np.nan)))
    if not frames:
        sys.exit("No slice folders matched SAMPLE_SHEET keys.")
    return pd.concat(frames, ignore_index=True).merge(sheet, on="key",
                                                      how="inner")


# =============================================================================
def fit_mixed(df, formula):
    """Random intercepts for animal, ovary within animal, ROI within ovary."""
    vc = {"ovary": "0 + C(ovary)", "roi": "0 + C(key)"}
    md = smf.mixedlm(formula, df, groups=df["animal"], vc_formula=vc,
                     re_formula="1")
    return md.fit(reml=True, method="lbfgs")


def report_variance(res):
    """Print the variance components.

    statsmodels warns 'MLE may be on the boundary' whenever a component is
    driven to zero, which is common with few animals. A component at zero
    means that level of nesting is contributing nothing and the effect above
    it is being tested with more degrees of freedom than the design supports.
    """
    parts = {k: v for k, v in res.params.items() if "Var" in k}
    if not parts:
        return
    print("\n  variance components (0 means that level collapsed):")
    for k, v in parts.items():
        label = {"Group Var": "animal"}.get(k, k.replace(" Var", ""))
        flag = "   <- collapsed to zero" if v < 1e-6 else ""
        print(f"    {label:<8s} {v:.6f}{flag}")


def contrast(res, di, cell_a, cell_b):
    """Difference between two cells of the age x compartment design.

    MixedLMResults.t_test wants a 2D matrix with exactly one column per fixed
    effect. res.params also carries the variance components, so the contrast
    has to be trimmed to the fixed effects and reshaped, not padded.
    """
    ra = np.asarray(dmatrix(di, pd.DataFrame([cell_a]))).ravel()
    rb = np.asarray(dmatrix(di, pd.DataFrame([cell_b]))).ravel()
    k = len(res.fe_params)
    if len(ra) != k:
        raise ValueError(f"design row has {len(ra)} terms but the model has "
                         f"{k} fixed effects; formula and design_info differ")
    t = res.t_test((ra - rb).reshape(1, k))
    return (float(np.ravel(t.effect)[0]),
            float(np.ravel(t.sd)[0]),
            float(np.ravel(t.pvalue)[0]))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj


# =============================================================================
def q1(df):
    print("\n" + "=" * 72)
    print("Q1  IS STIFFNESS DIFFERENT BETWEEN YOUNG AND AGED?")
    print("=" * 72)
    n_an = df.groupby("age")["animal"].nunique().to_dict()
    n_ov = df.groupby("age")["ovary"].nunique().to_dict()
    print(f"  indentations: {len(df):,}   "
          f"ovaries: {n_ov}   animals: {n_an}")

    res = fit_mixed(df, "log10E ~ C(age, Treatment('young'))")
    report_variance(res)
    name = [p for p in res.fe_params.index if "age" in p][0]
    eff, se, pv = res.fe_params[name], res.bse[name], res.pvalues[name]
    print(f"\n  Mixed model, aged vs young on log10 stiffness")
    print(f"    effect {eff:+.4f} log10 units  "
          f"({10**eff:.2f}-fold)   SE {se:.4f}   p = {pv:.4g}")
    print(f"    95% CI {10**(eff-1.96*se):.2f} to {10**(eff+1.96*se):.2f}-fold")

    ov = df.groupby(["ovary", "age"])["log10E"].mean().reset_index()
    y = ov.loc[ov["age"] == "young", "log10E"].values
    a = ov.loc[ov["age"] == "aged", "log10E"].values
    t, pt = ttest_ind(a, y)
    u, pu = mannwhitneyu(a, y, alternative="two-sided")
    print(f"\n  Per-ovary means, n = {len(y)} young vs {len(a)} aged")
    print(f"    young {', '.join(f'{10**v:.0f}' for v in sorted(y))} Pa")
    print(f"    aged  {', '.join(f'{10**v:.0f}' for v in sorted(a))} Pa")
    print(f"    t test        t = {t:+.3f}  p = {pt:.4g}")
    print(f"    Mann-Whitney  U = {u:.0f}  p = {pu:.4g}")
    print("\n  The three should agree. If the mixed model is far smaller than")
    print("  the other two, its variance components are unstable and the")
    print("  per-ovary tests are the ones to report.")
    return res


def q2(df):
    d = df.dropna(subset=["compartment"])
    d = d[d["compartment"].isin(COMPARTMENT_ORDER)].copy()
    if d.empty:
        print("\n[SKIP] Q2: no compartment labels.")
        return None
    print("\n" + "=" * 72)
    print("Q2  DO COMPARTMENTS DIFFER, WITHIN AND BETWEEN AGE GROUPS?")
    print("=" * 72)
    print(pd.crosstab(d["compartment"], d["age"]).to_string())

    f = ("log10E ~ C(age, Treatment('young')) * "
         "C(compartment, Treatment('S'))")
    res = fit_mixed(d, f)
    report_variance(res)
    di = dmatrix(f.split("~")[1].strip(), d, return_type="dataframe").design_info

    rows = []
    for age in ("young", "aged"):
        present = sorted(set(d.loc[d["age"] == age, "compartment"]),
                         key=COMPARTMENT_ORDER.index)
        for i, ca in enumerate(present):
            for cb in present[i + 1:]:
                e, s, p = contrast(res, di, dict(age=age, compartment=ca),
                                   dict(age=age, compartment=cb))
                rows.append(dict(family=f"within {age}", a=f"{age} {ca}",
                                 b=f"{age} {cb}", diff_log10=e, se=s, p=p))
    for c in COMPARTMENT_ORDER:
        if {("young", c), ("aged", c)} <= set(map(tuple,
                d[["age", "compartment"]].drop_duplicates().values)):
            e, s, p = contrast(res, di, dict(age="aged", compartment=c),
                               dict(age="young", compartment=c))
            rows.append(dict(family="aged vs young", a=f"aged {c}",
                             b=f"young {c}", diff_log10=e, se=s, p=p))

    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for fam, g in out.groupby("family"):
        out.loc[g.index, "p_holm"] = holm(g["p"].values)
    out["fold"] = 10 ** out["diff_log10"]
    out = out[["family", "a", "b", "diff_log10", "fold", "se", "p", "p_holm"]]
    print("\n" + out.to_string(index=False,
                               float_format=lambda v: f"{v:.4g}"))
    print("\n  p_holm is corrected within each family, not across all three.")
    return out


def write_prism(df):
    """One column per ovary x compartment, ragged, ready to paste into Prism.

    ROIs are collapsed, so every measurement from A6-1 and A6-2 that landed in
    stroma appears in a single A6-S column. Columns run young before aged, and
    CL, S, V, F within each ovary, which is the left-to-right order of the
    intended figure. Column lengths differ, so short columns are padded with
    blanks rather than zeros; Prism reads a blank cell as missing and a zero
    as a measurement of zero Pascals.
    """
    d = df.dropna(subset=["compartment"])
    d = d[d["compartment"].isin(COMPARTMENT_ORDER)]
    if d.empty:
        print("\n[SKIP] Prism export: no compartment labels.")
        return

    order = (d[["ovary", "age"]].drop_duplicates()
             .sort_values(["age", "ovary"], ascending=[False, True]))

    cols = {}
    for ov in order["ovary"]:
        for comp in COMPARTMENT_ORDER:
            v = d.loc[(d["ovary"] == ov) & (d["compartment"] == comp),
                      "stiffness"].dropna().values
            if len(v):
                cols[f"{ov}-{comp}"] = np.sort(v)

    n = max(len(v) for v in cols.values())
    wide = pd.DataFrame({k: np.concatenate([v, np.full(n - len(v), np.nan)])
                         for k, v in cols.items()})

    f_pa = OUT_DIR / "prism_stiffness_by_ovary_compartment_Pa.csv"
    f_kpa = OUT_DIR / "prism_stiffness_by_ovary_compartment_kPa.csv"
    wide.to_csv(f_pa, index=False)
    (wide / 1000.0).to_csv(f_kpa, index=False)

    print("\n" + "=" * 72)
    print("PRISM EXPORT")
    print("=" * 72)
    print(f"  {len(wide.columns)} columns, longest {n} values")
    counts = wide.notna().sum()
    for ov in order["ovary"]:
        age = d.loc[d["ovary"] == ov, "age"].iloc[0]
        bits = [f"{c.split('-')[1]} {counts[c]}" for c in wide.columns
                if c.startswith(f"{ov}-")]
        print(f"    {ov:<4s} ({age:>5s})  " + "   ".join(bits))
    print(f"\n  Pa  : {f_pa.name}")
    print(f"  kPa : {f_kpa.name}   (Fig. 2b axis is in kPa)")
    print("\n  Blank cells are missing values, not zeros. In Prism use a")
    print("  Grouped table and paste with 'Insert replicates down columns'.")


def write_prism_nested(df):
    """Block for a Prism Nested table: 8 groups x 4 ovary subcolumns.

    Groups run young CL, S, V, F then aged CL, S, V, F, left to right, which
    is the intended figure order. Within each group the four subcolumns are
    the four ovaries of that age, always in the same order, so an ovary with
    no measurements in that compartment leaves an empty subcolumn rather than
    shifting the others across. A2 has no corpus luteum measurements, so
    aged-CL keeps a blank third subcolumn.
    """
    d = df.dropna(subset=["compartment"])
    d = d[d["compartment"].isin(COMPARTMENT_ORDER)]
    if d.empty:
        print("\n[SKIP] nested export: no compartment labels.")
        return

    ov_by_age = {a: sorted(d.loc[d["age"] == a, "ovary"].unique())
                 for a in ("young", "aged")}

    cols, headers = {}, []
    for age in ("young", "aged"):
        for comp in COMPARTMENT_ORDER:
            for ov in ov_by_age[age]:
                v = d.loc[(d["ovary"] == ov) & (d["compartment"] == comp),
                          "stiffness"].dropna().values
                cols[f"{age}_{comp}_{ov}"] = np.sort(v)
                headers.append((f"{age} {comp}", ov, len(v)))

    n = max(len(v) for v in cols.values())
    wide = pd.DataFrame({k: np.concatenate([v, np.full(n - len(v), np.nan)])
                         for k, v in cols.items()})

    f = OUT_DIR / "prism_NESTED_stiffness_kPa.csv"
    (wide / 1000.0).to_csv(f, index=False)

    n_ov = len(ov_by_age["young"])
    print("\n" + "=" * 72)
    print("PRISM NESTED EXPORT")
    print("=" * 72)
    print(f"  {len(wide.columns)} columns = "
          f"{len(wide.columns)//n_ov} groups x {n_ov} subcolumns, "
          f"longest {n} values")
    for i in range(0, len(headers), n_ov):
        grp = headers[i][0]
        bits = "  ".join(f"{ov}:{k}" for _, ov, k in headers[i:i + n_ov])
        print(f"    {grp:<10s} {bits}")
    print(f"\n  {f.name}   (kPa, matching the Fig. 2b axis)")
    print("\n  In Prism: New table and graph, Nested, "
          f"{len(wide.columns)//n_ov} groups with {n_ov} subcolumns each,")
    print("  then paste the whole block into the first cell. Empty")
    print("  subcolumns are intentional and keep the groups aligned.")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    sheet = sheet_frame()
    df = load_points(sheet) if MODE == "points" else load_curves(sheet)
    df["log10E"] = np.log10(df["stiffness"])
    df = df.dropna(subset=["log10E"])
    print(f"MODE {MODE}   {len(df):,} observations")

    q1(df)
    tab = q2(df) if MODE == "points" else None

    write_prism(df)
    write_prism_nested(df)

    df.to_csv(OUT_DIR / "stiffness_observations.csv", index=False)
    if tab is not None:
        tab.to_csv(OUT_DIR / "compartment_contrasts.csv", index=False)
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()