"""
17_Extras_celltype_mechanics_regression.py

Elastic net regression predicting stiffness from cell type composition, with a
soft against stiff spatial comparison per compartment and per age group.

Run on this dataset it found little: composition explains almost none of the
variation in stiffness between measurements, which is the composition ceiling
reported in the manuscript and computed more directly by control C in script
16. The code is provided so it can be tried on tissues where composition and
mechanics may be more tightly coupled.

Parameters
    CLUSTER_COL         "CellCluster", the identity column used to build the
                        composition feature matrix
    PRIMARY_TARGET      "Elasticity_l_Pa"
    MIN_BINS_PER_POINT  5, measurements with fewer selected bins are excluded
    RANDOM_STATE        42
    Model               elastic net with leave-one-tissue-out cross-validation
                        and a permutation null
    Unit                the 50 um measurement. Bins are aggregated to their
                        measurement and the proportion of each cluster within
                        them becomes one row of the feature matrix. All ROIs
                        and tissues are pooled; there are no per-ROI outputs
    Runs are repeated globally, per compartment, and per compartment per age
    group, each writing to its own labelled subfolder

Actions required
    Set METADATA_TSV and CLUSTER_COL. Run script 9 first.
        py 17_Extras_celltype_mechanics_regression.py

Input
    metadata_with_nano.tsv                  from script 9
Output
    regressionoutputs/, as PNG, JPEG and SVG
        predicted against observed stiffness
        coefficients per cell type cluster
        young against aged predicted and observed
        spatial maps of every mechanical property
        soft against stiff spatial comparison
        PCA of the mechanical parameters
    Tables of regression coefficients, cross-validated performance and the
    permutation null.

Note
    No output from this script appears in the manuscript.

Dependencies
    pip install pandas numpy matplotlib seaborn scikit-learn scipy
"""

import re
import sys
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from textwrap import fill

from scipy.stats import norm as scipy_norm
import umap

warnings.filterwarnings("ignore")

# =============================================================================
# USER CONFIGURATION
# =============================================================================

METADATA_TSV = Path(r"Z:\Visium_Data\001_SpatialX\20260803_FinalAnnotatedMetadata\metadata_with_nano.tsv")
OUTPUT_DIR   = METADATA_TSV.parent / "regression_outputs"

# Cluster/cell-type label column.
CLUSTER_COL    = "CellCluster"

# Tissue compartment column
COMPARTMENT_COL = "Compartments"

# Run only these biologically meaningful compartments
COMPARTMENTS_TO_RUN = ["CL", "F", "S", "V"]

# Minimum number of nanoindentation points required after aggregation
MIN_NANO_POINTS_PER_COMPARTMENT = 10

RUN_AGE_STRATIFIED = True
AGE_STRATIFIED_COMPARTMENTS = ["CL", "F", "S", "V"]
MIN_NANO_POINTS_PER_AGE_SUBSET = 40
MIN_TISSUE_GROUPS_PER_AGE_SUBSET = 3

# Age group: function applied to the samples column to get "young" or "aged"
AGE_COL        = "AgeGroup"
AGE_LABEL_FN = lambda s: "young" if str(s).strip().lower() == "young" else "aged"

# Tissue identity column (used for leave-one-tissue-out cross-validation)
TISSUE_COL     = "Tissue-slices"

# Nanoindentation point grouping column
NANO_ID_COL    = "nano_point_id"

# ROI column (used to identify spatial stiffness region labelling)
ROI_COL        = "ROIs"

# Spatial coordinate columns
COL_X          = "x"
COL_Y          = "y"

# Mechanical targets -- a separate regression is run for each.
# Primary target is listed first.
TARGETS = [
    "Elasticity_l_Pa",
    "Elasticity_eff_Pa",
    "Loss",
    "Viscosity_Ns_m",
    "Energy_l_fJ",
    "Energy_ad_fJ",
]

PRIMARY_TARGET = "Elasticity_l_Pa"

MECH_UMAP_TARGETS = [
    "Elasticity_l_Pa",
    "Elasticity_eff_Pa",
    "Loss",
    "Viscosity_Ns_m",
    "Energy_l_fJ",
    "Energy_ad_fJ",
]

# Regression settings
MIN_BINS_PER_POINT = 5    # exclude nano points with fewer selected bins
N_PERMUTATIONS     = 2000  # permutations for null R2 distribution
RANDOM_STATE       = 42

# Plot colour scheme
YOUNG_COL  = "#007A7A"
AGED_COL   = "#B57BEE"
CMAP_STIFF = "YlOrRd"

SOFT_COL = "#56B4E9"
STIFF_COL = "#E69F00"

PLOT_DPI = 600
SVG_RASTERIZED_SCATTER = True

BASE_FONT = 12
TITLE_FONT = 15
SUBTITLE_FONT = 13
AXIS_FONT = 12
TICK_FONT = 10
LEGEND_FONT = 10

MECH_UMAP_TARGETS = [
    "Elasticity_l_Pa",
    "Elasticity_eff_Pa",
    "Loss",
    "Viscosity_Ns_m",
    "Energy_l_fJ",
    "Energy_ad_fJ",
]

def shifted_colormap(base_cmap, new_min_color, n_colors=256, shift_fraction=0.1):
    base = plt.get_cmap(base_cmap)
    base_colors = base(np.linspace(shift_fraction, 1, n_colors))
    new_min_color_rgba = to_rgba(new_min_color)
    new_colors = np.vstack((new_min_color_rgba, base_colors))
    return LinearSegmentedColormap.from_list(f"{base_cmap}_shifted", new_colors)

cmps = {
    "Z (um)": "Greys",
    "Energy_l (fJ)": "GnBu",
    "Loss": "RdYlBu",
    "Energy_ad (fJ)": "Blues",
    "Elasticity_l (Pa)": "YlOrRd",
    "Elasticity_eff (Pa)": "pink_r",
    "Viscosity (Ns/m)": "YlGn",
    "Power_l": "PuBuGn",
}

adjusted_cmps = {
    'Z (um)': shifted_colormap('Greys', '#bdbdbd'),
    'Energy_l (fJ)': shifted_colormap('YlGnBu', '#ccebc5'),
    'Loss': shifted_colormap('RdYlBu', '#f46d43'),
    'Energy_ad (fJ)': shifted_colormap('Blues', '#cde0f1'),
    'Elasticity_l (Pa)': shifted_colormap('YlOrRd', '#fee8c8'),
    'Elasticity_eff (Pa)': shifted_colormap('pink_r', '#f2f3d4'),
    'Viscosity (Ns/m)': shifted_colormap('YlGn', '#d9f0a3'),
    'Power_l': shifted_colormap('PuBuGn', '#d0d1e6'),
}

FEATURE_LIMITS = {
    "Elasticity_l_Pa": (0.0, 3000.0),
    "Elasticity_eff_Pa": (0.0, 5000.0),
    "Loss": (0.1, 0.7),
    "Viscosity_Ns_m": (0.0, 0.07),
    "Energy_l_fJ": (0.0, 65.0),
    "Energy_ad_fJ": (0.0, 36.0),
}

FEATURE_TICKS = {
    "Elasticity_l_Pa": [0, 750, 1500, 2250, 3000],
    "Elasticity_eff_Pa": [0, 1250, 2500, 3750, 5000],
    "Loss": [0.1, 0.25, 0.4, 0.55, 0.7],
    "Viscosity_Ns_m": [0, 0.0175, 0.035, 0.0525, 0.07],
    "Energy_l_fJ": [0, 16.25, 32.5, 48.75, 65],
    "Energy_ad_fJ": [0, 9, 18, 27, 36],
}

# =============================================================================
# LOGGING
# =============================================================================
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

# =============================================================================
# HELPERS
# =============================================================================
def save_figure(fig, stem, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["png", "jpg", "svg"]:
        path = out_dir / f"{stem}.{fmt}"
        kwargs = dict(bbox_inches="tight")
        if fmt in {"png", "jpg"}:
            kwargs["dpi"] = PLOT_DPI
        fig.savefig(path, format=fmt, **kwargs)
    print(f" Saved: {stem}.png / .jpg / .svg")

def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def clean_bool(series):
    return series.astype(str).str.strip().str.lower() == "true"

def standardize_tissue_group(val):
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    if s.lower() == "unassigned":
        return np.nan
    m = re.match(r"^([A-Za-z]+\d+)", s)
    return m.group(1) if m else s

def nice_target_label(target):
    mapping = {
        "Elasticity_l_Pa": r"$E$, Young's modulus (Pa)",
        "Elasticity_eff_Pa": r"$E_{eff}$ (Pa)",
        "Loss": r"$H$, hysteresis ratio",
        "Viscosity_Ns_m": "Viscosity (Ns/m)",
        "Energy_l_fJ": r"Energy$_{load}$ (fJ)",
        "Energy_ad_fJ": r"Energy$_{ad}$ (fJ)",
    }
    return mapping.get(target, target)

def colorbar_label(target):
    mapping = {
        "Elasticity_l_Pa": r"$E$ (Pa)",
        "Elasticity_eff_Pa": r"$E_{eff}$ (Pa)",
        "Loss": r"$H$, hysteresis ratio",
        "Viscosity_Ns_m": "Viscosity (Ns/m)",
        "Energy_l_fJ": r"Energy$_{load}$ (fJ)",
        "Energy_ad_fJ": r"Energy$_{ad}$ (fJ)",
    }
    return mapping.get(target, target)

def property_map_name(target):
    mapping = {
        "Elasticity_l_Pa": "Elasticity_l (Pa)",
        "Elasticity_eff_Pa": "Elasticity_eff (Pa)",
        "Loss": "Loss",
        "Viscosity_Ns_m": "Viscosity (Ns/m)",
        "Energy_l_fJ": "Energy_l (fJ)",
        "Energy_ad_fJ": "Energy_ad (fJ)",
    }
    return mapping.get(target, target)

def get_primary_result(results):
    for res in results:
        if res["target"] == PRIMARY_TARGET:
            return res
    return None

def setup_rcparams():
    plt.rcParams.update({
        "font.size": BASE_FONT,
        "axes.titlesize": SUBTITLE_FONT,
        "axes.labelsize": AXIS_FONT,
        "xtick.labelsize": TICK_FONT,
        "ytick.labelsize": TICK_FONT,
        "legend.fontsize": LEGEND_FONT,
        "figure.titlesize": TITLE_FONT,
    })

def sanitize_name(val):
    s = str(val).strip()
    s = re.sub(r"[^\w\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def filter_by_compartment(df, compartment_name):
    if COMPARTMENT_COL not in df.columns:
        raise ValueError(
            f"Compartment column '{COMPARTMENT_COL}' not found in metadata."
        )

    sub = df[
        df[COMPARTMENT_COL].notna() &
        (df[COMPARTMENT_COL].astype(str).str.strip() == str(compartment_name).strip())
    ].copy()

    print(f" Compartments kept = {compartment_name}: {len(sub):,} bins")
    return sub

def filter_by_age_group(agg, age_group):
    sub = agg[agg["age_group"] == age_group].copy()
    print(f" Age subset kept = {age_group}: {len(sub):,} nano points")
    return sub

# =============================================================================
# STEP 1: LOAD AND FILTER
# =============================================================================
def load_and_filter(path):
    print(f"Loading: {path}")
    df = pd.read_csv(path, sep='\t', low_memory=False)
    print(f" Total bins: {len(df):,}")

    required_cols = [
        CLUSTER_COL,
        COMPARTMENT_COL,
        AGE_COL,
        TISSUE_COL,
        ROI_COL,
        COL_X,
        COL_Y,
        NANO_ID_COL,
        "within_footprint",
        "valid_measurement",
        PRIMARY_TARGET,
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Input file is missing required columns for regression.\n"
            f"Missing: {missing}\n"
            f"File: {path}\n"
            "This script expects the output of the nanoindentation-mapping step "
            "(e.g. metadata_with_nano.tsv)."
        )

    df = df[
        clean_bool(df["within_footprint"]) &
        clean_bool(df["valid_measurement"])
    ].copy()
    print(f" Selected bins with valid measurements: {len(df):,}")

    before = len(df)
    df = df[df[ROI_COL].astype(str).str.strip().str.lower() != "unassigned"].copy()
    print(f" Excluded ROI='Unassigned': {before - len(df):,}")

    before = len(df)
    df = df[
        df[NANO_ID_COL].notna() &
        (df[NANO_ID_COL].astype(str).str.strip() != "N/A")
    ].copy()
    print(f" Bins with valid {NANO_ID_COL}: {len(df):,} (dropped {before - len(df):,})")

    before = len(df)
    df = df[
        df[CLUSTER_COL].notna() &
        (df[CLUSTER_COL].astype(str).str.strip() != "N/A")
    ].copy()
    print(f" Bins with valid {CLUSTER_COL}: {len(df):,} (dropped {before - len(df):,})")

    for col in TARGETS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[PRIMARY_TARGET]).copy()
    print(f" Bins with valid {PRIMARY_TARGET}: {len(df):,} (dropped {before - len(df):,})")

    df["age_group"] = df[AGE_COL].apply(AGE_LABEL_FN)
    df["tissue_group"] = df[TISSUE_COL].apply(standardize_tissue_group)

    before = len(df)
    df = df[df["tissue_group"].notna()].copy()
    print(f" Bins with assigned tissue_group: {len(df):,} (dropped {before - len(df):,})")

    print(f" Age groups: {df['age_group'].value_counts().to_dict()}")

    if "low_confidence" in df.columns:
        low_conf_n = clean_bool(df["low_confidence"]).sum()
        print(f" Low-confidence selected bins: {low_conf_n:,}")

    print(f" Unique nanoindentation points before aggregation: {df[NANO_ID_COL].nunique():,}")
    print(f" Tissue groups kept: {sorted(df['tissue_group'].dropna().unique().tolist())}")
    return df

# =============================================================================
# STEP 2: AGGREGATE TO NANOINDENTATION-POINT LEVEL
# =============================================================================

def aggregate_to_nano_points(df):
    print("\nAggregating to nanoindentation-point level...")

    cluster_counts = (
        df.groupby([NANO_ID_COL, CLUSTER_COL])
        .size()
        .unstack(fill_value=0)
    )

    cluster_props = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)
    cluster_props.columns = [f"cluster_{c}" for c in cluster_props.columns]

    agg_meta = pd.DataFrame(index=cluster_props.index)

    meta_first_cols = [TISSUE_COL, "tissue_group", AGE_COL, ROI_COL, "age_group"]
    if "low_confidence" in df.columns:
        meta_first_cols.append("low_confidence")

    for col in meta_first_cols:
        if col in df.columns:
            agg_meta[col] = df.groupby(NANO_ID_COL)[col].first()

    agg_meta["x_mean"] = df.groupby(NANO_ID_COL)[COL_X].mean()
    agg_meta["y_mean"] = df.groupby(NANO_ID_COL)[COL_Y].mean()
    agg_meta["n_bins"] = df.groupby(NANO_ID_COL)[CLUSTER_COL].count()

    for col in TARGETS:
        if col in df.columns:
            agg_meta[col] = df.groupby(NANO_ID_COL)[col].first()

    agg = agg_meta.join(cluster_props)

    before = len(agg)
    agg = agg[agg["n_bins"] >= MIN_BINS_PER_POINT].copy()
    print(f" Nano points: {len(agg):,} (excluded {before - len(agg):,} with < {MIN_BINS_PER_POINT} bins)")

    feature_cols = cluster_props.columns.tolist()
    print(f" Cell type features: {len(feature_cols)} clusters")
    print(f" Age groups: {agg['age_group'].value_counts().to_dict()}")
    print(f" Tissue groups: {agg['tissue_group'].nunique():,}")
    print(f" ROIs: {agg[ROI_COL].nunique():,}" if ROI_COL in agg.columns else " ROIs: n/a")
    return agg, feature_cols

# =============================================================================
# STEP 3: ELASTIC NET REGRESSION
# =============================================================================

def run_regression(agg, feature_cols, analysis_label="pooled"):
    results = []

    for target in TARGETS:
        if target not in agg.columns:
            print(f"\nSkipping {target}: column not in data.")
            continue

        df_v = agg[agg[target].notna()].copy()
        X_raw = df_v[feature_cols].fillna(0).values
        y = df_v[target].values.astype(float)
        tis = df_v["tissue_group"].values
        ages = df_v["age_group"].values

        unique_tissues = pd.unique(tis)
        n_tis = len(unique_tissues)
        print(f"\n--- {target} --- n={len(df_v):,} tissue_groups={n_tis}")

        min_points_required = MIN_NANO_POINTS_PER_AGE_SUBSET if analysis_label != "pooled" else 30
        min_tissues_required = MIN_TISSUE_GROUPS_PER_AGE_SUBSET if analysis_label != "pooled" else 2

        if len(df_v) < min_points_required or n_tis < min_tissues_required:
            print(f" Too few samples for {analysis_label}, skipping.")
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        gkf = GroupKFold(n_splits=n_tis)
        en = ElasticNetCV(
            l1_ratio=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
            cv=gkf.split(X, y, groups=tis),
            max_iter=20000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        en.fit(X, y)

        nonzero = np.where(en.coef_ != 0)[0]
        selected = [feature_cols[i] for i in nonzero]
        print(f" Selected features: {len(selected)} (alpha={en.alpha_:.5f}, l1_ratio={en.l1_ratio_:.2f})")

        if len(selected) == 0:
            print(" No features selected.")
            continue

        y_pred = np.full(len(y), np.nan, dtype=float)

        for test_tis in unique_tissues:
            tr = tis != test_tis
            te = tis == test_tis
            if tr.sum() < 5 or te.sum() == 0:
                continue
            ridge = Ridge(alpha=1.0)
            ridge.fit(X[tr][:, nonzero], y[tr])
            y_pred[te] = ridge.predict(X[te][:, nonzero])

        valid_pred = ~np.isnan(y_pred)
        if valid_pred.sum() < 2:
            print(" Too few valid CV predictions, skipping.")
            continue

        r2 = r2_score(y[valid_pred], y_pred[valid_pred])
        rmse = np.sqrt(mean_squared_error(y[valid_pred], y_pred[valid_pred]))
        print(f" CV R2={r2:.3f} RMSE={rmse:.2f}")

        rng = np.random.default_rng(RANDOM_STATE)
        null_r2s = []

        for _ in range(N_PERMUTATIONS):
            y_perm = y.copy()
            for t in unique_tissues:
                m = tis == t
                y_perm[m] = rng.permutation(y_perm[m])

            y_null = np.full(len(y_perm), np.nan, dtype=float)
            for test_tis in unique_tissues:
                tr = tis != test_tis
                te = tis == test_tis
                if tr.sum() < 5 or te.sum() == 0:
                    continue
                ridge = Ridge(alpha=1.0)
                ridge.fit(X[tr][:, nonzero], y_perm[tr])
                y_null[te] = ridge.predict(X[te][:, nonzero])

            vp = ~np.isnan(y_null)
            if vp.sum() > 1:
                null_r2s.append(r2_score(y_perm[vp], y_null[vp]))

        pval = float(np.mean(np.array(null_r2s) >= r2)) if null_r2s else np.nan
        print(f" Permutation p={pval:.4f}")

        coef_df = pd.DataFrame({
            "feature": selected,
            "coef": en.coef_[nonzero],
        }).sort_values("coef", key=np.abs, ascending=False)

        results.append({
            "target": target,
            "r2_cv": r2,
            "rmse": rmse,
            "pval_perm": pval,
            "null_r2s": null_r2s,
            "y_obs": y[valid_pred],
            "y_pred": y_pred[valid_pred],
            "age_group": ages[valid_pred],
            "tissue_group": tis[valid_pred],
            "coef_df": coef_df,
            "selected": selected,
            "analysis_label": analysis_label,
            "n_samples": int(valid_pred.sum()),
            "n_tissue_groups": int(len(pd.unique(tis))),
        })

    return results


# =============================================================================
# PLOT 1: PREDICTED VS OBSERVED
# =============================================================================

def plot_predicted_vs_observed(results, out_dir, label_suffix=None):
    print("\nPlot 01: Predicted vs observed...")

    primary_res = get_primary_result(results)
    if primary_res is None:
        print(f" No regression result found for {PRIMARY_TARGET}, skipping.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for group, col, marker in [("young", YOUNG_COL, "o"), ("aged", AGED_COL, "^")]:
        mask = primary_res["age_group"] == group
        if mask.sum() == 0:
            continue
        ax.scatter(
            primary_res["y_obs"][mask],
            primary_res["y_pred"][mask],
            c=col,
            marker=marker,
            s=36,
            alpha=0.65,
            linewidths=0,
            rasterized=SVG_RASTERIZED_SCATTER,
            label=group.capitalize()
        )

    x_lo = max(0, np.nanpercentile(primary_res["y_obs"], 1))
    x_hi = np.nanpercentile(primary_res["y_obs"], 99)
    y_lo = max(0, np.nanpercentile(primary_res["y_pred"], 1))
    y_hi = np.nanpercentile(primary_res["y_pred"], 99)

    line_lo = min(x_lo, y_lo)
    line_hi = max(x_hi, y_hi)

    ax.plot([line_lo, line_hi], [line_lo, line_hi], "k--", lw=1.2, alpha=0.5)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    pval_str = "p<0.001" if primary_res["pval_perm"] < 0.001 else f"p={primary_res['pval_perm']:.3f}"
    ax.set_title(
        f"CV R² = {primary_res['r2_cv']:.3f}, {pval_str}",
        fontsize=SUBTITLE_FONT
    )

    fig.suptitle(
        r"Predicted vs observed $E$, Young's modulus (Pa)",
        fontsize=TITLE_FONT, y=0.98
    )

    stem = "plot01_predicted_vs_observed"
    if label_suffix:
        stem += f"_{sanitize_name(label_suffix)}"
    save_figure(fig, stem, out_dir)


# =============================================================================
# PLOT 2: FEATURE COEFFICIENTS
# =============================================================================

def plot_coefficients(results, out_dir, label_suffix=None):
    print("\nPlot 02: Feature coefficients...")

    if len(results) == 0:
        print(" No coefficient results to plot.")
        return

    n = len(results)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(5.5 * nrows, 4.0)))
    fig.subplots_adjust(left=0.22, right=0.98, top=0.90, bottom=0.10, wspace=0.40, hspace=0.30)
    axes = np.array(axes).reshape(-1)

    for idx, res in enumerate(results):
        ax = axes[idx]
        cdf = res["coef_df"].copy()

        if cdf.empty:
            ax.set_visible(False)
            continue

        cdf = cdf.sort_values("coef", key=np.abs, ascending=True)
        display = (
            cdf["feature"]
            .str.replace("cluster_", "", regex=False)
            .apply(lambda s: fill(str(s), width=20))
        )
        colors = [YOUNG_COL if v > 0 else AGED_COL for v in cdf["coef"]]

        ax.barh(
            range(len(cdf)),
            cdf["coef"].values,
            color=colors,
            alpha=0.85,
            height=0.55,
            rasterized=SVG_RASTERIZED_SCATTER if len(cdf) > 40 else False
        )
        ax.axvline(0, color="k", lw=1.0, alpha=0.4)
        ax.set_yticks(range(len(cdf)))
        ax.set_yticklabels(display, fontsize=TICK_FONT, ha="right")
        ax.set_xlabel("Coefficient (standardised)", fontsize=AXIS_FONT)
        ax.set_title(f"{nice_target_label(res['target'])}", fontsize=SUBTITLE_FONT)
        ax.tick_params(labelsize=TICK_FONT)
        despine(ax)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Cell type cluster coefficients from elastic net regression",
        fontsize=TITLE_FONT,
        y=0.98
    )
    stem = "plot02_feature_coefficients"
    if label_suffix:
        stem += f"_{sanitize_name(label_suffix)}"
    save_figure(fig, stem, out_dir)
    plt.close(fig)


# =============================================================================
# PLOT 3: YOUNG VS AGED COMPARISON
# =============================================================================

def plot_young_vs_aged(results, out_dir, label_suffix=None):
    print("\nPlot 03: Young vs aged comparison...")

    primary_res = get_primary_result(results)
    if primary_res is None:
        print(f" No regression result found for {PRIMARY_TARGET}, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for col_idx, (group, color, marker) in enumerate([("young", YOUNG_COL, "o"), ("aged", AGED_COL, "^")]):
        ax = axes[col_idx]
        mask = primary_res["age_group"] == group
        if mask.sum() < 2:
            ax.set_title(f"{group.capitalize()} (no data)", fontsize=SUBTITLE_FONT)
            ax.set_visible(True)
            continue

        yo = primary_res["y_obs"][mask]
        yp = primary_res["y_pred"][mask]
        r2g = r2_score(yo, yp)
        n_g = mask.sum()

        ax.scatter(
            yo, yp,
            c=color,
            marker=marker,
            s=36,
            alpha=0.65,
            linewidths=0,
            rasterized=SVG_RASTERIZED_SCATTER
        )

        all_vals = np.r_[yo, yp]
        lo = max(0, np.nanpercentile(all_vals, 1))
        hi = np.nanpercentile(all_vals, 99)

        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        ax.set_title(f"{group.capitalize()}\nR²={r2g:.3f} n={n_g}", fontsize=SUBTITLE_FONT, color=color)
        ax.set_xlabel(f"Observed {nice_target_label(primary_res['target'])}", fontsize=AXIS_FONT)
        ax.set_ylabel(f"Predicted {nice_target_label(primary_res['target'])}", fontsize=AXIS_FONT)
        ax.tick_params(labelsize=TICK_FONT)
        despine(ax)

    fig.suptitle("Young vs aged: primary stiffness prediction", fontsize=TITLE_FONT, y=0.995)

    stem = "plot03_young_vs_aged"
    if label_suffix:
        stem += f"_{sanitize_name(label_suffix)}"
    save_figure(fig, stem, out_dir)
    plt.close(fig)


# =============================================================================
# SAVE TABLES
# =============================================================================

def save_tables(results, out_dir, label_suffix=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{sanitize_name(label_suffix)}" if label_suffix else ""

    perf = pd.DataFrame([{
        "target": r["target"],
        "n_features_selected": len(r["selected"]),
        "r2_cv": round(r["r2_cv"], 4),
        "rmse": round(r["rmse"], 4),
        "permutation_p": round(r["pval_perm"], 4),
        "analysis_label": r.get("analysis_label", "pooled"),
        "n_samples": r.get("n_samples", np.nan),
        "n_tissue_groups": r.get("n_tissue_groups", np.nan),
    } for r in results])
    perf_name = f"model_performance{suffix}.csv"
    perf.to_csv(out_dir / perf_name, index=False)
    print(f"\nSaved: {perf_name}")

    coef_rows = []
    for r in results:
        for _, row in r["coef_df"].iterrows():
            coef_rows.append({
                "target": r["target"],
                "feature": row["feature"],
                "coef": round(float(row["coef"]), 6),
                "analysis_label": r.get("analysis_label", "pooled"),
            })
    coef_name = f"coefficients{suffix}.csv"
    pd.DataFrame(coef_rows).to_csv(out_dir / coef_name, index=False)

    pred_rows = []
    for r in results:
        for yo, yp, ag, ti in zip(r["y_obs"], r["y_pred"], r["age_group"], r["tissue_group"]):
            pred_rows.append({
                "target": r["target"],
                "observed": round(float(yo), 4),
                "predicted": round(float(yp), 4),
                "age_group": ag,
                "tissue_group": ti,
                "analysis_label": r.get("analysis_label", "pooled"),
            })
    pred_name = f"predicted_vs_observed{suffix}.csv"
    pd.DataFrame(pred_rows).to_csv(out_dir / pred_name, index=False)

    print(f"Saved: {coef_name}, {pred_name}")


# =============================================================================
# PLOT 4: MECHANICS-ONLY PCA
# =============================================================================

def plot_mechanics_pca(agg, out_dir, label_suffix=None):
    print("\nPlot 04: Mechanics-only PCA...")

    avail = [t for t in MECH_UMAP_TARGETS if t in agg.columns]
    if len(avail) < 2:
        print(" Fewer than 2 mechanical properties available, skipping.")
        return

    df_plot = agg.dropna(subset=["age_group", "tissue_group"]).copy()
    if len(df_plot) < 3:
        print(" Too few points after filtering age/tissue labels, skipping.")
        return

    X_raw = df_plot[avail].values.astype(float)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_raw)
    X_sc = StandardScaler().fit_transform(X_imp)

    n_pc = min(X_sc.shape[1], X_sc.shape[0], 4)
    if n_pc < 2:
        print(" Too few points for PCA, skipping.")
        return

    pca = PCA(n_components=n_pc, random_state=RANDOM_STATE)
    scores = pca.fit_transform(X_sc)

    pc1_var = 100 * pca.explained_variance_ratio_[0]
    pc2_var = 100 * pca.explained_variance_ratio_[1]

    df_plot["PC1"] = scores[:, 0]
    df_plot["PC2"] = scores[:, 1]

    cvals = df_plot[PRIMARY_TARGET].astype(float).values
    vmin, vmax = FEATURE_LIMITS.get(
        PRIMARY_TARGET,
        (float(np.nanmin(cvals)), float(np.nanmax(cvals)))
    )
    
    unique_tissues = sorted(pd.unique(df_plot["tissue_group"]))
    tissue_palette = {
        tis: col for tis, col in zip(
            unique_tissues,
            ["#328ec1", "#009E73", "#CC79A7", "#332288",
             "#004f7c", "#0C682A", "#882255", "#1F1066"][:len(unique_tissues)]
        )
    }

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 7.2))
    fig.subplots_adjust(left=0.12, right=0.78, top=0.88, bottom=0.12)

    for age_group, marker in [("young", "o"), ("aged", "^")]:
        sub = df_plot[df_plot["age_group"] == age_group].copy()
        if len(sub) == 0:
            continue

        edgecols = [tissue_palette[t] for t in sub["tissue_group"]]
        sc = ax.scatter(
            sub["PC1"],
            sub["PC2"],
            c=sub[PRIMARY_TARGET].clip(lower=vmin, upper=vmax),
            cmap="Greys",
            vmin=vmin,
            vmax=vmax,
            s=52,
            marker=marker,
            alpha=0.95,
            edgecolors=edgecols,
            linewidths=1.2,
            rasterized=SVG_RASTERIZED_SCATTER
        )

    ax.set_xlabel(f"PC1 ({pc1_var:.1f}% variance)", fontsize=AXIS_FONT)
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}% variance)", fontsize=AXIS_FONT)
    ax.tick_params(labelsize=TICK_FONT)
    despine(ax)

    fig.suptitle(
        "Mechanics-only PCA",
        fontsize=TITLE_FONT,
        y=0.96
    )

    ax.set_title("Shape: age | outline: tissue slice", fontsize=SUBTITLE_FONT, pad=10)

    cax = fig.add_axes([0.88, 0.12, 0.025, 0.33])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.ax.tick_params(labelsize=TICK_FONT)
    cbar.ax.set_title(colorbar_label(PRIMARY_TARGET), fontsize=AXIS_FONT, pad=8)

    tissue_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor=col,
            markeredgewidth=1.5,
            color=col,
            markersize=7,
            label=tis
        )
        for tis, col in tissue_palette.items()
    ]

    ax.legend(
        handles=tissue_handles,
        title="Tissue slice",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.78),
        frameon=False,
        fontsize=LEGEND_FONT,
        title_fontsize=LEGEND_FONT
    )

    stem = "plot04_mechanics_pca"
    if label_suffix:
        stem += f"_{sanitize_name(label_suffix)}"
    save_figure(fig, stem, out_dir)
    plt.close(fig)


# =============================================================================
# PLOT 5: SPATIAL MAPS: ALL MECHANICAL PROPERTIES
# =============================================================================

def plot_spatial_all_properties(df, out_dir, label_suffix=None):
    print("\nPlot 05: Spatial maps for all mechanical properties...")

    avail = [t for t in MECH_UMAP_TARGETS if t in df.columns]
    if not avail:
        print(" No mechanical property columns found, skipping.")
        return

    young_groups = sorted(df.loc[df["age_group"] == "young", "tissue_group"].dropna().unique())
    aged_groups = sorted(df.loc[df["age_group"] == "aged", "tissue_group"].dropna().unique())

    nrows = max(len(young_groups), len(aged_groups))
    if nrows == 0:
        print(" No tissue groups available, skipping.")
        return

    for prop in avail:
        sub_all = df.dropna(subset=[prop]).copy()
        if len(sub_all) == 0:
            continue

        vmin, vmax = FEATURE_LIMITS.get(
            prop,
            (float(sub_all[prop].min()), float(sub_all[prop].max()))
        )
        sub_all["_plot_value"] = sub_all[prop].clip(lower=vmin, upper=vmax)

        map_prop = property_map_name(prop)
        this_cmap = adjusted_cmps.get(map_prop, cmps.get(map_prop, CMAP_STIFF))


        fig = plt.figure(figsize=(10, max(3.2 * nrows, 8)))
        gs = fig.add_gridspec(
            nrows=nrows,
            ncols=2,
            wspace=0.10,
            hspace=0.70
        )

        last_sc = None

        for i in range(nrows):
            # left column = young
            ax_left = fig.add_subplot(gs[i, 0])
            if i < len(young_groups):
                grp = young_groups[i]
                sub = sub_all[
                    (sub_all["tissue_group"] == grp) &
                    (sub_all["age_group"] == "young")
                ].copy()

                if len(sub) > 0:                    
                    last_sc = ax_left.scatter(
                        sub[COL_X], sub[COL_Y],
                        c=sub["_plot_value"],
                        cmap=this_cmap,
                        vmin=vmin,
                        vmax=vmax,
                        s=1.2,
                        alpha=0.9,
                        linewidths=0,
                        rasterized=SVG_RASTERIZED_SCATTER
                    )
                    ax_left.set_aspect("equal")
                    mean_val = sub[prop].mean()
                    sd_val = sub[prop].std()

                    ax_left.set_title(f"{grp}", fontsize=SUBTITLE_FONT, pad=6)
                    ax_left.set_xlabel("x (µm)", fontsize=AXIS_FONT)
                    ax_left.set_ylabel("y (µm)", fontsize=AXIS_FONT)
                    ax_left.tick_params(labelsize=TICK_FONT)
                    despine(ax_left)
                else:
                    ax_left.set_visible(False)
            else:
                ax_left.set_visible(False)

            # right column = aged
            ax_right = fig.add_subplot(gs[i, 1])
            if i < len(aged_groups):
                grp = aged_groups[i]
                sub = sub_all[
                    (sub_all["tissue_group"] == grp) &
                    (sub_all["age_group"] == "aged")
                ].copy()

                if len(sub) > 0:
                    last_sc = ax_right.scatter(
                        sub[COL_X], sub[COL_Y],
                        c=sub["_plot_value"],
                        cmap=this_cmap,
                        vmin=vmin,
                        vmax=vmax,
                        s=1.2,
                        alpha=0.9,
                        linewidths=0,
                        rasterized=SVG_RASTERIZED_SCATTER
                    )
                    ax_right.set_aspect("equal")
                    mean_val = sub[prop].mean()
                    sd_val = sub[prop].std()

                    ax_right.set_title(f"{grp}", fontsize=SUBTITLE_FONT, pad=6)
                    ax_right.set_xlabel("x (µm)", fontsize=AXIS_FONT)
                    ax_right.set_ylabel("y (µm)", fontsize=AXIS_FONT)
                    ax_right.tick_params(labelsize=TICK_FONT)
                    despine(ax_right)
                else:
                    ax_right.set_visible(False)
            else:
                ax_right.set_visible(False)
        fig.text(0.28, 0.945, "Young", ha="center", va="center", fontsize=SUBTITLE_FONT)
        fig.text(0.72, 0.945, "Aged", ha="center", va="center", fontsize=SUBTITLE_FONT)

        cax = fig.add_axes([0.90, 0.35, 0.03, 0.50])

        if last_sc is not None:
            cbar = fig.colorbar(last_sc, cax=cax)
            cbar.ax.tick_params(labelsize=TICK_FONT)

            ticks = FEATURE_TICKS.get(prop)
            if ticks is not None:
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([str(t) for t in ticks])

            cbar.ax.set_title(
                colorbar_label(prop),
                fontsize=AXIS_FONT,
                pad=8
            )

        fig.suptitle(
            f"Spatial distribution of {nice_target_label(prop)}\n",
            fontsize=TITLE_FONT,
            y=0.97
        )

        stem = f"plot05_spatial_{prop}"
        if label_suffix:
            stem += f"_{sanitize_name(label_suffix)}"
        save_figure(fig, stem, out_dir)
        plt.close(fig)

    print(f" Saved spatial maps for {len(avail)} properties (png / jpg / svg each)")


# Global Mechanics

def plot_spatial_all_properties_global(df, out_dir, label_suffix=None):
    print("\nPlot 05b: Global spatial maps for all mechanical properties (no compartment split)...")

    avail = [t for t in MECH_UMAP_TARGETS if t in df.columns]
    if not avail:
        print(" No mechanical property columns found, skipping.")
        return

    young_groups = sorted(df.loc[df["age_group"] == "young", "tissue_group"].dropna().unique())
    aged_groups = sorted(df.loc[df["age_group"] == "aged", "tissue_group"].dropna().unique())

    nrows = max(len(young_groups), len(aged_groups))
    if nrows == 0:
        print(" No tissue groups available, skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for prop in avail:
        sub_all = df.dropna(subset=[prop]).copy()
        if len(sub_all) == 0:
            continue

        vmin, vmax = FEATURE_LIMITS.get(
            prop,
            (float(sub_all[prop].min()), float(sub_all[prop].max()))
        )
        sub_all["_plot_value"] = sub_all[prop].clip(lower=vmin, upper=vmax)

        map_prop = property_map_name(prop)
        this_cmap = adjusted_cmps.get(map_prop, cmps.get(map_prop, CMAP_STIFF))

        fig = plt.figure(figsize=(10, max(3.2 * nrows, 8)))
        gs = fig.add_gridspec(
            nrows=nrows,
            ncols=2,
            wspace=0.10,
            hspace=0.70
        )

        last_sc = None

        for i in range(nrows):
            ax_left = fig.add_subplot(gs[i, 0])
            if i < len(young_groups):
                grp = young_groups[i]
                sub = sub_all[
                    (sub_all["tissue_group"] == grp) &
                    (sub_all["age_group"] == "young")
                ].copy()

                if len(sub) > 0:
                    last_sc = ax_left.scatter(
                        sub[COL_X], sub[COL_Y],
                        c=sub["_plot_value"],
                        cmap=this_cmap,
                        vmin=vmin,
                        vmax=vmax,
                        s=1.2,
                        alpha=0.9,
                        linewidths=0,
                        rasterized=SVG_RASTERIZED_SCATTER
                    )
                    ax_left.set_aspect("equal")
                    mean_val = sub[prop].mean()
                    sd_val = sub[prop].std()
                    ax_left.set_title(f"{grp}", fontsize=SUBTITLE_FONT, pad=6)
                    ax_left.set_xlabel("x (µm)", fontsize=AXIS_FONT)
                    ax_left.set_ylabel("y (µm)", fontsize=AXIS_FONT)
                    ax_left.tick_params(labelsize=TICK_FONT)
                    despine(ax_left)
                else:
                    ax_left.set_visible(False)
            else:
                ax_left.set_visible(False)

            ax_right = fig.add_subplot(gs[i, 1])
            if i < len(aged_groups):
                grp = aged_groups[i]
                sub = sub_all[
                    (sub_all["tissue_group"] == grp) &
                    (sub_all["age_group"] == "aged")
                ].copy()

                if len(sub) > 0:
                    last_sc = ax_right.scatter(
                        sub[COL_X], sub[COL_Y],
                        c=sub["_plot_value"],
                        cmap=this_cmap,
                        vmin=vmin,
                        vmax=vmax,
                        s=1.2,
                        alpha=0.9,
                        linewidths=0,
                        rasterized=SVG_RASTERIZED_SCATTER
                    )
                    ax_right.set_aspect("equal")
                    mean_val = sub[prop].mean()
                    sd_val = sub[prop].std()
                    ax_right.set_title(f"{grp}", fontsize=SUBTITLE_FONT, pad=6)
                    ax_right.set_xlabel("x (µm)", fontsize=AXIS_FONT)
                    ax_right.set_ylabel("y (µm)", fontsize=AXIS_FONT)
                    ax_right.tick_params(labelsize=TICK_FONT)
                    despine(ax_right)
                else:
                    ax_right.set_visible(False)
            else:
                ax_right.set_visible(False)
        fig.text(0.28, 0.945, "Young", ha="center", va="center", fontsize=SUBTITLE_FONT)
        fig.text(0.72, 0.945, "Aged", ha="center", va="center", fontsize=SUBTITLE_FONT)
        
        cax = fig.add_axes([0.90, 0.35, 0.03, 0.50])

        if last_sc is not None:
            cbar = fig.colorbar(last_sc, cax=cax)
            cbar.ax.tick_params(labelsize=TICK_FONT)

            ticks = FEATURE_TICKS.get(prop)
            if ticks is not None:
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([str(t) for t in ticks])

            cbar.ax.set_title(
                colorbar_label(prop),
                fontsize=AXIS_FONT,
                pad=8
            )

        fig.suptitle(
            f"Global spatial distribution of {nice_target_label(prop)}\n",
            fontsize=TITLE_FONT,
            y=0.97
        )

        stem = f"plot05b_spatial_{prop}"
        if label_suffix:
            stem += f"_{sanitize_name(label_suffix)}"
        save_figure(fig, stem, out_dir)
        plt.close(fig)

    print(f" Saved global spatial maps for {len(avail)} properties (png / jpg / svg each)")

# =============================================================================
# PLOT 6: SOFT/STIFF COMPARTMENT MAP
# =============================================================================

def plot_soft_stiff_spatial(agg, out_dir, label_suffix=None):
    print("\nPlot 06: Soft/stiff compartment GMM fit...")

    valid = agg[PRIMARY_TARGET].notna() & (agg[PRIMARY_TARGET] > 0)
    agg_v = agg[valid].copy()
    agg_v["log_target"] = np.log10(agg_v[PRIMARY_TARGET].astype(float))
    y = agg_v["log_target"].values

    if len(y) < 10:
        print(" Too few valid points, skipping.")
        return

    gmm2 = GaussianMixture(n_components=2, random_state=RANDOM_STATE)
    gmm1 = GaussianMixture(n_components=1, random_state=RANDOM_STATE)

    gmm2.fit(y.reshape(-1, 1))
    gmm1.fit(y.reshape(-1, 1))

    probs = gmm2.predict_proba(y.reshape(-1, 1))
    means = gmm2.means_.flatten()
    soft_ci = int(np.argmin(means))
    stiff_ci = int(np.argmax(means))

    soft_log = float(gmm2.means_[soft_ci, 0])
    stiff_log = float(gmm2.means_[stiff_ci, 0])
    soft_var = float(gmm2.covariances_[soft_ci, 0, 0])
    stiff_var = float(gmm2.covariances_[stiff_ci, 0, 0])
    soft_sd = np.sqrt(soft_var)
    stiff_sd = np.sqrt(stiff_var)
    w_soft = float(gmm2.weights_[soft_ci])
    w_stiff = float(gmm2.weights_[stiff_ci])

    soft_pa = 10 ** soft_log
    stiff_pa = 10 ** stiff_log

    bic2 = gmm2.bic(y.reshape(-1, 1))
    bic1 = gmm1.bic(y.reshape(-1, 1))
    bimodal = bic2 < bic1
    fit_label = "bimodal" if bimodal else "unimodal-preferred"

    x_lo = min(y.min(), soft_log - 4 * soft_sd, stiff_log - 4 * stiff_sd)
    x_hi = max(y.max(), soft_log + 4 * soft_sd, stiff_log + 4 * stiff_sd)
    x_grid = np.linspace(x_lo, x_hi, 2000)

    d_soft = w_soft * scipy_norm.pdf(x_grid, soft_log, soft_sd)
    d_stiff = w_stiff * scipy_norm.pdf(x_grid, stiff_log, stiff_sd)
    d_mix = d_soft + d_stiff

    post_grid = gmm2.predict_proba(x_grid.reshape(-1, 1))
    p_soft_grid = post_grid[:, soft_ci]
    p_stiff_grid = post_grid[:, stiff_ci]

    probs_soft = probs[:, soft_ci]
    probs_stiff = probs[:, stiff_ci]
    max_probs = np.maximum(probs_soft, probs_stiff)

    thresholds = [0.7, 0.8, 0.9]
    thr_colors = {0.7: "#66C2A5", 0.8: "#FC8D62", 0.9: "#8DA0CB"}
    thr_results = []

    for thr in thresholds:
        soft_mask = p_soft_grid >= thr
        stiff_mask = p_stiff_grid >= thr

        x_soft_thr = np.nan
        x_stiff_thr = np.nan

        if np.any(soft_mask):
            x_soft_thr = x_grid[np.where(soft_mask)[0].max()]
        if np.any(stiff_mask):
            x_stiff_thr = x_grid[np.where(stiff_mask)[0].min()]

        keep_mask = max_probs >= thr
        soft_keep = (probs_soft >= thr)
        stiff_keep = (probs_stiff >= thr)

        thr_results.append({
            "thr": thr,
            "x_soft": x_soft_thr,
            "x_stiff": x_stiff_thr,
            "kept_n": int(keep_mask.sum()),
            "kept_pct": 100.0 * keep_mask.mean(),
            "soft_n": int(soft_keep.sum()),
            "stiff_n": int(stiff_keep.sum())
        })

        print(
            f" Confidence >= {thr:.1f}: kept {keep_mask.sum()}/{len(agg_v)} "
            f"({100.0 * keep_mask.mean():.1f}%) | soft={soft_keep.sum()}, stiff={stiff_keep.sum()}"
        )

    agg_v["p_soft"] = probs_soft
    agg_v["p_stiff"] = probs_stiff
    agg_v["p_max"] = max_probs
    agg_v["compartment_gmm"] = np.where(probs_stiff > probs_soft, "stiff", "soft")

    print(f" Soft component mean: log10={soft_log:.3f}, ~{soft_pa:.1f} Pa")
    print(f" Stiff component mean: log10={stiff_log:.3f}, ~{stiff_pa:.1f} Pa")
    print(f" Component weights: soft={w_soft:.3f}, stiff={w_stiff:.3f}")
    print(
        f" Model-selection scores (BIC; lower is better for the same dataset because "
        f"it indicates the preferred balance of fit and model complexity): "
        f"BIC(2) = {bic2:.1f} vs BIC(1) = {bic1:.1f} -> "
        f"{'2-component preferred' if bimodal else '1-component preferred'}"
    )

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 6.6))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.12)

    sns.histplot(
        y,
        bins=30,
        stat="density",
        color="lightgrey",
        edgecolor="#555555",
        linewidth=1.0,
        ax=ax
    )

    ax.plot(x_grid, d_soft, color=SOFT_COL, lw=2.2, label="soft Gaussian")
    ax.plot(x_grid, d_stiff, color=STIFF_COL, lw=2.2, label="stiff Gaussian")
    ax.plot(x_grid, d_mix, color="black", lw=1.6, ls="--", alpha=0.8, label="mixture")

    legend_handles = [
        Line2D([0], [0], color=SOFT_COL, lw=2.2, label="soft Gaussian"),
        Line2D([0], [0], color=STIFF_COL, lw=2.2, label="stiff Gaussian"),
        Line2D([0], [0], color="black", lw=1.6, ls="--", label="mixture"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fontsize=9,
        title=f"BIC: {'2-comp preferred' if bimodal else '1-comp preferred'}",
        title_fontsize=9
    )

    for res in thr_results:
        thr = res["thr"]
        c = thr_colors[thr]

        if np.isfinite(res["x_soft"]):
            mask_soft = x_grid <= res["x_soft"]
            ax.fill_between(
                x_grid[mask_soft], 0, d_soft[mask_soft],
                color=SOFT_COL, alpha=0.08 + (thr - 0.7) * 0.18
            )
            ax.axvline(res["x_soft"], color=c, lw=1.6, ls="--", alpha=0.95)

        if np.isfinite(res["x_stiff"]):
            mask_stiff = x_grid >= res["x_stiff"]
            ax.fill_between(
                x_grid[mask_stiff], 0, d_stiff[mask_stiff],
                color=STIFF_COL, alpha=0.08 + (thr - 0.7) * 0.18
            )
            ax.axvline(res["x_stiff"], color=c, lw=1.6, ls="--", alpha=0.95)

    for i, res in enumerate(thr_results):
        txt = (
            f"{res['thr']:.1f}: kept {res['kept_n']}/{len(agg_v)} ({res['kept_pct']:.1f}%)"
            f"\nsoft={res['soft_n']} stiff={res['stiff_n']}"
        )
        ax.text(
            0.03,
            0.97 - i * 0.085,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color=thr_colors[res["thr"]],
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="white",
                edgecolor="none",
                alpha=0.80
            )
        )

    ax.set_xlabel(f"log10 {nice_target_label(PRIMARY_TARGET)}", fontsize=AXIS_FONT)
    ax.set_ylabel("Density", fontsize=AXIS_FONT)
    title_label = label_suffix if label_suffix is not None else "all"

    ax.set_title(
        f"GMM fit ({title_label}): {fit_label}\n"
        f"soft ~{int(round(soft_pa))} Pa vs stiff ~{int(round(stiff_pa))} Pa",
        fontsize=SUBTITLE_FONT,
        pad=10
    )

    fig.subplots_adjust(top=0.90)

    stem = f"plot06_{sanitize_name(title_label)}_soft_stiff_spatial"
    save_figure(fig, stem, out_dir)
    plt.close(fig)

    save_cols = [
        PRIMARY_TARGET, "log_target", "p_soft", "p_stiff", "p_max",
        "compartment_gmm", "age_group", "tissue_group", ROI_COL, "x_mean", "y_mean"
    ]
    agg_v[save_cols].to_csv(
        out_dir / f"plot06_{sanitize_name(title_label)}_soft_stiff_compartment_assignments.csv",
        index=True
    )
    print(f" Saved: plot06_{sanitize_name(title_label)}_soft_stiff_compartment_assignments.csv")


# =============================================================================
# MAIN
# =============================================================================

def main():
    setup_rcparams()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_DIR / f"run_log_{timestamp}.txt"

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)

        try:
            print("=" * 70)
            print("Cell type-mechanics regression run")
            print(f"Date/time: {datetime.now().isoformat(timespec='seconds')}")
            print(f"Metadata TSV: {METADATA_TSV}")
            print(f"Output directory: {OUTPUT_DIR}")
            print(f"Cluster column: {CLUSTER_COL}")
            print(f"Compartment column: {COMPARTMENT_COL}")
            print(f"Compartments to run: {COMPARTMENTS_TO_RUN}")
            print(f"Age column: {AGE_COL}")
            print(f"Tissue column: {TISSUE_COL}")
            print(f"Nano point column: {NANO_ID_COL}")
            print(f"ROI column: {ROI_COL}")
            print(f"Primary target: {PRIMARY_TARGET}")
            print(f"Targets: {TARGETS}")
            print(f"MIN_BINS_PER_POINT: {MIN_BINS_PER_POINT}")
            print(f"MIN_NANO_POINTS_PER_COMPARTMENT: {MIN_NANO_POINTS_PER_COMPARTMENT}")
            print(f"N_PERMUTATIONS: {N_PERMUTATIONS}")
            print(f"RANDOM_STATE: {RANDOM_STATE}")
            print(f"Run log: {log_path}")
            print("=" * 70)

            df = load_and_filter(METADATA_TSV)

            agg_global, feature_cols_global = aggregate_to_nano_points(df)
            plot_mechanics_pca(agg_global, OUTPUT_DIR, label_suffix="global")

            if COMPARTMENT_COL not in df.columns:
                raise ValueError(
                    f"Required compartment column '{COMPARTMENT_COL}' not found in metadata."
                )

            all_compartments = sorted(
                df[COMPARTMENT_COL].dropna().astype(str).str.strip().unique().tolist()
            )
            print(f" Compartments present after filtering: {all_compartments}")

            summary_rows = []

            plot_spatial_all_properties_global(df, OUTPUT_DIR, label_suffix="global")

            for comp_name in COMPARTMENTS_TO_RUN:
                print("\n" + "=" * 70)
                print(f"RUNNING COMPARTMENT ANALYSIS: {comp_name}")

                df_run = filter_by_compartment(df, comp_name)

                out_dir = OUTPUT_DIR
                label_suffix = sanitize_name(comp_name)

                if len(df_run) == 0:
                    print(" No bins after filtering, skipping.")
                    summary_rows.append({
                        "compartment": comp_name,
                        "status": "skipped_no_bins",
                        "n_bins": 0,
                        "n_nano_points": 0,
                        "n_features": 0
                    })
                    continue

                agg, feature_cols = aggregate_to_nano_points(df_run)

                if len(agg) < MIN_NANO_POINTS_PER_COMPARTMENT:
                    print(
                        f" Too few nano points after aggregation ({len(agg)}), "
                        f"skipping analysis for {comp_name}."
                    )
                    summary_rows.append({
                        "compartment": comp_name,
                        "status": "skipped_too_few_nano_points",
                        "n_bins": len(df_run),
                        "n_nano_points": len(agg),
                        "n_features": len(feature_cols)
                    })
                    continue

                print("\nRunning elastic net regression...")
                results = run_regression(agg, feature_cols)

                plot_predicted_vs_observed(results, out_dir, label_suffix=label_suffix)
                plot_coefficients(results, out_dir, label_suffix=label_suffix)
                plot_young_vs_aged(results, out_dir, label_suffix=label_suffix)
                plot_spatial_all_properties(df_run, out_dir, label_suffix=label_suffix)
                plot_soft_stiff_spatial(agg, out_dir, label_suffix=comp_name)

                if results:
                    save_tables(results, out_dir, label_suffix=label_suffix)
                    for r in results:
                        summary_rows.append({
                            "compartment": comp_name,
                            "status": "ok",
                            "n_bins": len(df_run),
                            "n_nano_points": len(agg),
                            "n_features": len(r["selected"]),
                            "target": r["target"],
                            "r2_cv": r["r2_cv"],
                            "pval_perm": r["pval_perm"]
                        })
                else:
                    summary_rows.append({
                        "compartment": comp_name,
                        "status": "no_models_returned",
                        "n_bins": len(df_run),
                        "n_nano_points": len(agg),
                        "n_features": len(feature_cols)
                    })
                    
                if RUN_AGE_STRATIFIED and comp_name in AGE_STRATIFIED_COMPARTMENTS:
                    for age_group in ["young", "aged"]:
                        print("\n" + "=" * 70)
                        print(f"RUNNING AGE-STRATIFIED ANALYSIS: {comp_name} | {age_group}")
                        print("=" * 70)

                        agg_age = filter_by_age_group(agg, age_group)

                        if len(agg_age) < MIN_NANO_POINTS_PER_AGE_SUBSET:
                            print(f" Skipping {comp_name} | {age_group}: too few nano points.")
                            continue

                        n_tis_age = agg_age["tissue_group"].nunique()
                        if n_tis_age < MIN_TISSUE_GROUPS_PER_AGE_SUBSET:
                            print(f" Skipping {comp_name} | {age_group}: too few tissue groups ({n_tis_age}).")
                            continue

                        age_out_dir = OUTPUT_DIR
                        age_label_suffix = f"{sanitize_name(comp_name)}_{age_group}"

                        print(f"\nRunning elastic net regression ({age_group})...")
                        age_results = run_regression(
                            agg_age,
                            feature_cols,
                            analysis_label=age_group
                        )

                        if age_results:
                            save_tables(age_results, age_out_dir, label_suffix=age_label_suffix)
                            plot_predicted_vs_observed(age_results, age_out_dir, label_suffix=age_label_suffix)
                            plot_coefficients(age_results, age_out_dir, label_suffix=age_label_suffix)
                            plot_soft_stiff_spatial(agg_age, age_out_dir, label_suffix=f"{comp_name}-{age_group}")

                            for r in age_results:
                                summary_rows.append({
                                    "compartment": comp_name,
                                    "analysis_label": age_group,
                                    "status": "ok",
                                    "n_bins": len(df_run),
                                    "n_nano_points": len(agg_age),
                                    "n_features": len(r["selected"]),
                                    "target": r["target"],
                                    "r2_cv": r["r2_cv"],
                                    "pval_perm": r["pval_perm"]
                                })
                        else:
                            summary_rows.append({
                                "compartment": comp_name,
                                "analysis_label": age_group,
                                "status": "no_models_returned",
                                "n_bins": len(df_run),
                                "n_nano_points": len(agg_age),
                                "n_features": len(feature_cols)
                            })

                print(f"\nCompleted analysis for: {comp_name}")
                print(f"Outputs saved to: {out_dir}")

            print("\n" + "=" * 70)
            print("COMPARTMENT REGRESSION SUMMARY")
            print("=" * 70)

            summary_df = pd.DataFrame(summary_rows)
            if len(summary_df) > 0:
                summary_df.to_csv(OUTPUT_DIR / "compartment_regression_summary.csv", index=False)
                print(summary_df)
                print(f"\nSaved: {OUTPUT_DIR / 'compartment_regression_summary.csv'}")
            else:
                print("No compartment analyses were completed.")

            print("\nOutputs generated per compartment:")
            print(" - Spatial stiffness maps")
            print(" - Predicted vs observed plot (primary target)")
            print(" - Feature coefficient plots")
            print(" - Young vs aged comparison plot (primary target)")
            print(" - Spatial maps for all mechanical properties")
            print(" - Mechanics-only PCA")
            print(" - Soft/stiff spatial compartment figure")
            print(" - Regression summary tables (CSV), where available")

            print(f"\nTop-level outputs saved to: {OUTPUT_DIR}")
            print("=" * 70)
            print(f"Run log saved to: {log_path}")

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

if __name__ == "__main__":
    main()