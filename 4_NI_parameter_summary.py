"""
4_NI_parameter_summary.py

Summarises every mechanical parameter the pipeline emits, per slice and per
age group. Script 3 counts how many points succeeded; this reports what they
measured.

Parameters
    VALID_MIN, VALID_MAX  7.6 to 22,726 Pa, plausible range for E_l
    PERR_MAX              0.15, Hertzian fit relative standard error
    VISCOSITY_CAP         1.0 N s m^-1
    ONE_SIDED_PARAMS      {"Elasticity_l"}, one-sided because aged > young was
                          the prior hypothesis. Everything else is two-sided
    A point failing QC is dropped for every parameter, so all parameters
    describe the same set of measurements.

Actions required
    Set INCLUDE_ONLY to the slice labels carrying matched spatial
    transcriptomic data. Left empty, every slice folder is included, which is
    not the cohort reported in the manuscript.
    Place in the project root, one folder per slice beneath it.
        py 4_NI_parameter_summary.py

Input
    <root>/<slice>/output/NI_<slice>_all_samples.csv     from script 1
                                                         ("all samples.csv" is
                                                         used as a fallback)
Output
    <root>/NI_parameter_summary_per_slice.csv   n, median, IQR, 5th and 95th
                                                percentile and geometric mean
                                                per slice per parameter
    <root>/NI_parameter_summary_per_group.csv   the same pooled by age group,
                                                with per-ovary medians and a
                                                Mann-Whitney on those medians

Note
    The unit is the ovary, not the indentation. With 4 versus 4 the smallest
    possible two-sided p is 0.029, so a parameter can differ substantially and
    still not reach significance. Read effect size before p.
"""

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# =============================================================================
# CONFIG
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent   # folder holding the slice folders

# Slice labels to include. Empty list = every folder found, which will pull in
# ovaries that have no matched transcriptome. Fill this with the 8 reported.
INCLUDE_ONLY = []          # e.g. ["Y1-1", "Y1-2", "Y2-2", "Y4-1", ...]

VALID_MIN, VALID_MAX = 7.6, 22726.0          # Pa, plausible range for E_l
PERR_MAX = 0.15                              # Hertzian fit relative SE
VISCOSITY_CAP = 1.0                          # N s m^-1

# One-sided was used for Young's modulus (aged > young was the prior
# hypothesis). Everything else has no prior direction, so two-sided.
ONE_SIDED_PARAMS = {"Elasticity_l"}

ENCODINGS = ["utf-8-sig", "cp1252", "latin-1"]

OUT_PER_SLICE = ROOT_DIR / "NI_parameter_summary_per_slice.csv"
OUT_PER_GROUP = ROOT_DIR / "NI_parameter_summary_per_group.csv"

# Identifiers and flags, not measured parameters.
NOT_PARAMETERS = {"x", "y", "index", "point", "indentationpresent",
                  "status", "file", "slice", "csv", "name", "id"}
ERROR_COL_PATTERN = re.compile(r"^perr", re.IGNORECASE)

# Trailing slice label in a folder name, e.g. "..._young-Y1-1" -> "Y1-1"
LABEL_PATTERN = re.compile(r"([YA]\d+(?:-\d+)?)\s*$", re.IGNORECASE)


# =============================================================================
def canon(col):
    """'Elasticity_l (Pa)' -> 'Elasticity_l'."""
    c = re.sub(r"\s*\([^)]*\)\s*$", "", str(col)).strip()
    return c.replace(" ", "_")


def unit_of(col):
    m = re.search(r"\(([^)]*)\)\s*$", str(col))
    return m.group(1) if m else ""


def geometric_mean(v):
    """Geometric mean, or NaN if any value is non-positive (Loss, adhesion)."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0 or np.any(v <= 0):
        return np.nan
    return float(np.exp(np.mean(np.log(v))))


def read_csv_any(path):
    """Read with the first encoding that works. Pipeline headers carry a
    degree sign, which is not valid UTF-8 when written on Windows."""
    last = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last = e
    raise last


def pick_csv(folder):
    """One canonical CSV per slice folder, or None.

    Resource forks are skipped outright: '._all samples.csv' is a metadata
    stub, not data, and reading it silently would corrupt every number
    downstream.
    """
    out = folder / "output"
    if not out.is_dir():
        return None, []
    cands = [p for p in sorted(out.glob("*.csv"))
             if not p.name.startswith("._")
             and "all" in p.name.lower() and "samples" in p.name.lower()]
    if not cands:
        return None, []
    preferred = out / f"NI_{folder.name}_all_samples.csv"
    if preferred in cands:
        return preferred, cands
    plain = [p for p in cands if p.name.lower() == "all samples.csv"]
    if plain:
        return plain[0], cands
    return None, cands


def parse_folder(name):
    """Return (label, age_group). Label is None when the folder carries no
    Y#/A# suffix, which is how the extra ovaries are told apart."""
    low = name.lower()
    if "young" in low:
        age = "young"
    elif "aged" in low or "old" in low:
        age = "aged"
    else:
        age = "unknown"
    m = LABEL_PATTERN.search(name)
    return (m.group(1).upper() if m else None), age


def describe(v):
    v = pd.to_numeric(pd.Series(v), errors="coerce").dropna()
    if v.empty:
        return dict(n=0, median=np.nan, q25=np.nan, q75=np.nan,
                    p5=np.nan, p95=np.nan, geomean=np.nan)
    return dict(n=int(v.size), median=float(v.median()),
                q25=float(v.quantile(0.25)), q75=float(v.quantile(0.75)),
                p5=float(v.quantile(0.05)), p95=float(v.quantile(0.95)),
                geomean=geometric_mean(v))


def load_slice(label, age, path):
    d = read_csv_any(path)
    d.columns = [c.strip() for c in d.columns]
    units = {canon(c): unit_of(c) for c in d.columns}

    # Stripping the unit can make two distinct headers collide, e.g.
    # 'Elasticity_l (Pa)' and 'Elasticity_l (kPa)', and a duplicated name
    # makes d[name] return a DataFrame rather than a Series. Keep the first
    # occurrence and say which ones collided, rather than silently averaging
    # two different quantities into one column.
    new_names = [canon(c) for c in d.columns]
    seen, collisions = {}, []
    for orig, new in zip(d.columns, new_names):
        if new in seen:
            collisions.append((seen[new], orig, new))
        else:
            seen[new] = orig
    d.columns = new_names
    if collisions:
        for first, dup, new in collisions:
            print(f"      [DUP] {label}: '{first}' and '{dup}' both become "
                  f"'{new}'; keeping '{first}'")
        d = d.loc[:, ~pd.Index(new_names).duplicated(keep="first")]

    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    n_raw = len(d)
    if "Elasticity_l" not in d.columns:
        return None, units, n_raw, 0, "no Elasticity_l column"

    mask = d["Elasticity_l"].between(VALID_MIN, VALID_MAX, inclusive="both")
    note = ""
    if "Perr_l" in d.columns:
        mask &= d["Perr_l"] <= PERR_MAX
    else:
        note = "no Perr_l, fit-quality filter skipped"
    d = d[mask].copy()

    if "Viscosity" in d.columns:
        d.loc[d["Viscosity"].abs() > VISCOSITY_CAP, "Viscosity"] = np.nan

    d["slice"] = label
    d["age_group"] = age
    return d, units, n_raw, len(d), note


def parameter_columns(d):
    out = []
    for c in d.columns:
        if c in ("slice", "age_group") or c.lower() in NOT_PARAMETERS:
            continue
        if ERROR_COL_PATTERN.match(c):
            continue
        if pd.api.types.is_numeric_dtype(d[c]):
            out.append(c)
    return out


def main():
    print("=" * 72)
    print(f"ROOT: {ROOT_DIR}")
    if INCLUDE_ONLY:
        print(f"INCLUDE_ONLY: {', '.join(INCLUDE_ONLY)}")
    else:
        print("INCLUDE_ONLY is empty: every slice folder will be included.")
        print("  This is probably NOT the cohort you want to report.")
    print("=" * 72)

    folders = [p for p in sorted(ROOT_DIR.iterdir())
               if p.is_dir() and not p.name.startswith(".")]

    frames, units_all, skipped, unlabelled = [], {}, [], []
    for f in folders:
        label, age = parse_folder(f.name)
        if label is None:
            unlabelled.append(f.name)
            continue
        if INCLUDE_ONLY and label not in INCLUDE_ONLY:
            continue
        path, cands = pick_csv(f)
        if path is None:
            skipped.append((f.name, f"no usable CSV among {len(cands)}"))
            continue
        d, units, n_raw, n_keep, note = load_slice(label, age, path)
        units_all.update({k: v for k, v in units.items() if v})
        if d is None or d.empty:
            skipped.append((f.name, note or "empty after QC"))
            continue
        frames.append(d)
        print(f"  {label:<7s} {age:>5s}  {n_keep:>5,} of {n_raw:>5,} pass QC"
              f"   <- {path.name}" + (f"   [{note}]" if note else ""))

    if unlabelled:
        print(f"\n  [SKIPPED, no Y#/A# label] {len(unlabelled)} folder(s):")
        for n in unlabelled:
            print(f"      {n}")
        print("  Add a label to the folder name if any of these belong.")
    if skipped:
        print(f"\n  [SKIPPED] {len(skipped)}:")
        for n, why in skipped:
            print(f"      {n}: {why}")

    if not frames:
        sys.exit("\nNothing loaded. Check INCLUDE_ONLY and the folder names.")

    all_d = pd.concat(frames, ignore_index=True)
    params = parameter_columns(all_d)

    print("\n" + "=" * 72)
    print(f"PARAMETERS DETECTED ({len(params)})")
    print("=" * 72)
    for p in params:
        u = units_all.get(p, "")
        print(f"  {p}" + (f"  [{u}]" if u else "  [no unit in header]"))
    print("\n  Cross-check against the number of plots the pipeline writes.")
    print("  A mismatch means the column list and the plotting list have")
    print("  drifted apart.")

    per_slice = []
    for (sl, ag), g in all_d.groupby(["slice", "age_group"]):
        for p in params:
            row = dict(slice=sl, age_group=ag, parameter=p,
                       unit=units_all.get(p, ""))
            row.update(describe(g[p]))
            per_slice.append(row)
    per_slice = pd.DataFrame(per_slice)
    per_slice.to_csv(OUT_PER_SLICE, index=False)

    per_group = []
    for p in params:
        sub = per_slice[per_slice["parameter"] == p]
        row = dict(parameter=p, unit=units_all.get(p, ""))
        for ag in ("young", "aged"):
            for k, v in describe(all_d.loc[all_d["age_group"] == ag, p]).items():
                row[f"{ag}_{k}"] = v
            per_ov = sub.loc[sub["age_group"] == ag, "median"].dropna().values
            row[f"{ag}_n_slices"] = len(per_ov)
            row[f"{ag}_slice_medians"] = ", ".join(f"{x:.4g}"
                                                   for x in sorted(per_ov))
        y = sub.loc[sub["age_group"] == "young", "median"].dropna().values
        a = sub.loc[sub["age_group"] == "aged", "median"].dropna().values
        if len(y) >= 2 and len(a) >= 2:
            alt = "less" if p in ONE_SIDED_PARAMS else "two-sided"
            stat, pv = mannwhitneyu(y, a, alternative=alt)
            row["nested_test"] = f"Mann-Whitney, {alt}, n={len(y)} vs {len(a)}"
            row["nested_U"], row["nested_p"] = float(stat), float(pv)
        else:
            row["nested_test"] = "not run, fewer than 2 slices per group"
            row["nested_U"] = row["nested_p"] = np.nan
        per_group.append(row)
    per_group = pd.DataFrame(per_group)
    per_group.to_csv(OUT_PER_GROUP, index=False)

    print("\n" + "=" * 72)
    print("PER AGE GROUP  (pooled over all retained indentations)")
    print("=" * 72)
    for _, r in per_group.iterrows():
        u = f" {r['unit']}" if r["unit"] else ""
        print(f"\n{r['parameter']}{u}")
        for ag in ("young", "aged"):
            print(f"    {ag:>5s}  n={int(r[f'{ag}_n']):>6,}   "
                  f"median {r[f'{ag}_median']:>10.4g}   "
                  f"IQR {r[f'{ag}_q25']:.4g} to {r[f'{ag}_q75']:.4g}")
            print(f"           per-slice medians: {r[f'{ag}_slice_medians']}")
        if np.isfinite(r["nested_p"]):
            print(f"    nested: p = {r['nested_p']:.4g}   ({r['nested_test']})")
        else:
            print(f"    nested: {r['nested_test']}")

    print("\n" + "=" * 72)
    print(f"Written:\n  {OUT_PER_SLICE}\n  {OUT_PER_GROUP}")
    print("\nWith 4 slices per group the smallest two-sided p is 0.029, so a")
    print("parameter can differ substantially and still not reach p < 0.05.")
    print("Quote medians and per-slice values, not only the p.")


if __name__ == "__main__":
    main()