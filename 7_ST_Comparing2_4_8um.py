"""
7_ST_Comparing2_4_8um.py

Compares Visium HD binned at 2, 4 and 8 um on the same sections, to justify
the bin size used for every downstream analysis. Runs QC, clustering and
spatial mapping at each resolution.

Parameters
    BIN_SIZES     [2, 4, 8]
    N_PCS         20
    CLUSTER_RES   0.6, the same for all three so cluster counts are comparable
    RANDOM_SEED   42
    SUBSAMPLE     150,000 bins at 2 and 4 um. Bin counts at those resolutions
                  exceed what can be clustered directly, so clustering runs on
                  the subsample and labels are projected back to all bins by
                  kNN in PC space, chunked at PROJ_CHUNK to stay in memory.
                  Spatial coherence is therefore assessed on the full tissue
    RASTER_DPI    600

Actions required
    Set CA_DIRS and SLICE_NAMES to the four capture areas. Runs one capture
    area at a time with all cores, leaving 4 for the OS.
        py 7_ST_Comparing2_4_8um.py

Input
    Space Ranger output per capture area at 2, 4 and 8 um: count matrices,
    spatial coordinates and metadata
Output
    Per capture area per resolution, as PNG and SVG:
        UMI counts per bin, genes per bin, and their spatial distribution
        UMAP and spatial cluster maps
        PC standard deviation
    These are the panels in Supplementary Fig. 3.

Dependencies
    pip install scanpy anndata h5py pillow pyarrow scipy pandas matplotlib
                pynndescent leidenalg louvain python-igraph cmcrameri

Note
    Cluster number at a fixed resolution is not monotonic in bin size. The
    excess at 2 um reflects sparsity rather than biology, and the apparent
    deficit at 4 um relative to 8 um is an artifact of the subsampling step.
    Neither count is comparable to the 38 populations reported in the
    manuscript, which came from the integrated four-capture-area object at
    resolution 0.8 in script 8.
"""

# ── Thread counts set before ANY import so numba/openblas pick them up ──────
# Sequential mode: all available cores go to the single active CA
import os as _os
import multiprocessing as _mp

_N_CORES        = _mp.cpu_count()
_THREADS_PER_CA = max(1, _N_CORES - 4)   # leave 4 cores headroom for OS

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    _os.environ[_var] = str(_THREADS_PER_CA)

import gc
import os
import json
import time
import warnings
import multiprocessing
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
mpl.rcParams["font.family"] = "Arial"

warnings.filterwarnings("ignore")

sc.settings.verbosity = 1
sc.settings.n_jobs    = _THREADS_PER_CA

# =============================================================================
# USER CONFIGURATION
# =============================================================================
CA_DIRS = {
    "CA_01": r"Z:\Visium_Data\HTO-01-IF-001",
    "CA_02": r"Z:\Visium_Data\HTO-02-IF-001",
    "CA_03": r"Z:\Visium_Data\HTO-03-IF-001",
    "CA_04": r"Z:\Visium_Data\HTO-04-IF-001",
}

SLICE_NAMES = {
    "CA_01": "HTO-01",
    "CA_02": "HTO-02",
    "CA_03": "HTO-03",
    "CA_04": "HTO-04",
}

BIN_SIZES     = [2, 4, 8]
N_PCS         = 20
CLUSTER_RES   = 0.6
RANDOM_SEED   = 42
# Subsample sizes for clustering — labels projected back to full dataset for spatial plots
# Set to None to cluster full dataset (fast enough for small bin counts)
SUBSAMPLE = {
    2: 150_000,   # 4.3M bins → 150k
    4: 150_000,   # 1.1M bins → 150k
    8: None,      # ~120k bins → cluster in full
}
RASTER_DPI    = 600
PROJ_CHUNK    = 50_000   # bins per chunk during kNN label projection (~600 MB peak per chunk)

# Dot size (matplotlib s = points²) per resolution at 600 DPI
# Formula: s = (px * 72 / 600)² → 4px=0.2304, 8px=0.4608, 16px=0.9216
# 2um gets 4px, 4um=8px, 8um=16px
DOT_SIZE = {2: 0.2304, 4: 0.4608, 8: 0.9216}

# =============================================================================
# UNIFIED FONT SIZES
# =============================================================================
FS = {
    "title"    : 12,
    "axis_label": 12,
    "tick"     : 10,
    "legend"   : 10,
    "colorbar" : 9,
    "annot"    : 8,
}

# =============================================================================
# FIGURE SAVING
# =============================================================================
def save_fig_plain(fig, path_no_ext, dpi_png=600):
    fig.savefig(path_no_ext + ".png", dpi=dpi_png, bbox_inches="tight")
    fig.savefig(path_no_ext + ".svg", dpi=dpi_png, bbox_inches="tight")
    plt.close(fig)

def save_fig_rasterised(fig, path_no_ext, dpi_png=600):
    """
    PNG at full DPI + SVG where scatter artists (rasterized=True) are embedded
    as a rasterised image at dpi_png, while axes, labels, title and legend
    remain as selectable vector elements.
    Requires svglite or matplotlib's built-in svg backend — no extra packages.
    """
    fig.savefig(path_no_ext + ".png", dpi=dpi_png, bbox_inches="tight")
    fig.savefig(path_no_ext + ".svg", dpi=dpi_png, bbox_inches="tight", format="svg")
    plt.close(fig)

# =============================================================================
# DATA LOADING
# =============================================================================
def load_visiumhd_bin(base_dir, bin_um, slice_name):
    import h5py
    import scipy.sparse as sp
    import anndata as ad
    from PIL import Image

    bin_str     = f"{bin_um:03d}"
    bin_dir     = Path(base_dir) / "binned_outputs" / f"square_{bin_str}um"
    spatial_dir = bin_dir / "spatial"

    h5_path = bin_dir / "filtered_feature_bc_matrix.h5"
    with h5py.File(str(h5_path), "r") as f:
        grp        = f["matrix"]
        barcodes   = grp["barcodes"][:].astype(str)
        gene_ids   = grp["features"]["id"][:].astype(str)
        gene_names = grp["features"]["name"][:].astype(str)
        data       = grp["data"][:]
        indices    = grp["indices"][:]
        indptr     = grp["indptr"][:]
        shape      = tuple(grp["shape"][:])

    X      = sp.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()
    var_df = pd.DataFrame({"gene_ids": gene_ids, "gene_symbols": gene_names},
                          index=gene_names)
    obs_df = pd.DataFrame(index=barcodes)
    adata  = ad.AnnData(X=X, obs=obs_df, var=var_df)
    adata.var_names_make_unique()

    # Spatial positions
    pos_parquet = spatial_dir / "tissue_positions.parquet"
    pos_csv     = spatial_dir / "tissue_positions.csv"
    pos_csv_old = spatial_dir / "tissue_positions_list.csv"

    if pos_parquet.exists():
        pos = pd.read_parquet(str(pos_parquet))
        if "barcode" in pos.columns:
            pos = pos.set_index("barcode")
    elif pos_csv.exists():
        pos = pd.read_csv(str(pos_csv), index_col=0)
    elif pos_csv_old.exists():
        pos = pd.read_csv(str(pos_csv_old), header=None, index_col=0,
                          names=["in_tissue","array_row","array_col",
                                 "pxl_row_in_fullres","pxl_col_in_fullres"])
    else:
        raise FileNotFoundError(f"No tissue positions file in {spatial_dir}")

    pos = pos.reindex(barcodes)
    adata.obsm["spatial"] = pos[["pxl_col_in_fullres",
                                  "pxl_row_in_fullres"]].values.astype(float)
    adata.obs["in_tissue"] = pos["in_tissue"].values
    adata.obs["array_row"] = pos["array_row"].values
    adata.obs["array_col"] = pos["array_col"].values

    with open(str(spatial_dir / "scalefactors_json.json")) as f:
        scalef = json.load(f)
    img_path = spatial_dir / "tissue_hires_image.png"
    img_key  = "hires"
    if not img_path.exists():
        img_path = spatial_dir / "tissue_lowres_image.png"
        img_key  = "lowres"
    img = np.array(Image.open(str(img_path)))
    adata.uns["spatial"] = {
        slice_name: {"images": {img_key: img}, "scalefactors": scalef}
    }
    adata.uns["slice_name"] = slice_name
    adata.obs["bin_um"]     = bin_um

    adata = adata[adata.obs["in_tissue"] == 1].copy()
    print(f"    Loaded: {adata.n_obs:,} bins x {adata.n_vars:,} genes")
    return adata

# =============================================================================
# COLOUR HELPERS
# =============================================================================
def _ca_slug(ca_label):
    return ca_label.lower().replace("_", "")


def _spatial_coords(adata):
    coords = adata.obsm["spatial"]
    return coords[:, 0], coords[:, 1]


def _batlow_colours(labels):
    """batlow: perceptually uniform, colour-blind friendly, B&W-readable."""
    try:
        import cmcrameri.cm as cmc
        cmap = cmc.batlow
    except ImportError:
        print("    WARNING: cmcrameri not installed — using tab20. "
              "Run: pip install cmcrameri")
        cmap = plt.get_cmap("tab20")
    unique    = sorted(labels.unique(), key=lambda x: int(x))
    positions = np.linspace(0, 1, len(unique))
    return {lbl: cmap(pos) for lbl, pos in zip(unique, positions)}

# =============================================================================
# PLOT FUNCTIONS
# =============================================================================
def plot_qc_violin(adata, bin_um, ca_label, fig_dir):
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    fig, axes = plt.subplots(1, 2, figsize=(3, 3))
    for ax, key, label in zip(
        axes,
        ["total_counts", "n_genes_by_counts"],
        [f"nCount_{bin_um:03d}µm", f"nFeature_{bin_um:03d}µm"],
    ):
        ax.violinplot(adata.obs[key].values, showmedians=True, widths=0.8)
        ax.set_xticks([1])
        ax.set_xticklabels([f"{ca_label} {bin_um:03d}µm"], fontsize=FS["tick"])
        ax.set_ylabel(label, fontsize=FS["axis_label"])
        ax.tick_params(axis="y", labelsize=FS["tick"])
    fig.suptitle(f"{ca_label} {bin_um}µm QC",
                 fontsize=FS["title"], fontweight="bold")
    fig.tight_layout()
    save_fig_plain(fig, os.path.join(fig_dir,
        f"{_ca_slug(ca_label)}_{bin_um:02d}µm_QC_vln"), dpi_png=600)

def plot_spatial_ncount(adata, bin_um, ca_label, fig_dir):
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    x, y   = _spatial_coords(adata)
    counts = adata.obs["total_counts"].values
    vmax   = float(np.percentile(counts, 95))

    dot_size = DOT_SIZE.get(bin_um, 0.0144)

    fig, ax = plt.subplots(figsize=(6, 6))
    sc_plot = ax.scatter(x, y, c=counts, s=dot_size, alpha=1.0,
                         cmap="viridis", vmin=0, vmax=vmax,
                         marker=".", linewidths=0, edgecolors="none",
                         rasterized=True)
    cbar = fig.colorbar(sc_plot, ax=ax, shrink=0.7)
    cbar.set_label(f"UMI count (95th pct = {vmax:.0f})", fontsize=FS["colorbar"])
    cbar.ax.tick_params(labelsize=FS["tick"])
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"{ca_label} {bin_um}µm — nCount",
                 fontsize=FS["title"], fontweight="bold")
    ax.set_xlabel("x (µm)", fontsize=FS["axis_label"])
    ax.set_ylabel("y (µm)", fontsize=FS["axis_label"])
    ax.tick_params(labelsize=FS["tick"])
    fig.tight_layout()
    save_fig_rasterised(fig, os.path.join(fig_dir,
        f"{_ca_slug(ca_label)}_{bin_um:02d}µm_spatial_nCount"), dpi_png=600)

def plot_elbow(adata, bin_um, ca_label, fig_dir):
    std_devs = np.sqrt(adata.uns["pca"]["variance"][:N_PCS])
    fig, ax  = plt.subplots(figsize=(3, 3))
    ax.scatter(range(1, len(std_devs) + 1), std_devs,
               s=30, color="black", zorder=3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel("PC", fontsize=FS["axis_label"])
    ax.set_ylabel("Standard Deviation", fontsize=FS["axis_label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.set_title(f"{ca_label} {bin_um}um — Elbow",
                 fontsize=FS["title"], fontweight="bold")
    fig.tight_layout()
    save_fig_plain(fig, os.path.join(fig_dir,
        f"{_ca_slug(ca_label)}_{bin_um:02d}um_elbowplot"), dpi_png=600)

def plot_umap_clusters(adata, bin_um, ca_label, fig_dir):
    umap       = adata.obsm["X_umap"]
    labels     = adata.obs["leiden"]
    colour_map = _batlow_colours(labels)
    fig, ax    = plt.subplots(figsize=(6, 6))
    for lbl, col in colour_map.items():
        mask = (labels == lbl).values
        ax.scatter(umap[mask, 0], umap[mask, 1],
                   c=[col], s=0.9216, alpha=1.0, label=lbl,
                   marker=".", linewidths=0, edgecolors="none",
                   rasterized=True)
    for lbl, col in colour_map.items():
        mask = (labels == lbl).values
        ax.text(float(umap[mask, 0].mean()), float(umap[mask, 1].mean()),
                lbl, fontsize=FS["annot"], ha="center", va="center",
                fontweight="bold", color="black")
    ax.set_xlabel("UMAP_1", fontsize=FS["axis_label"])
    ax.set_ylabel("UMAP_2", fontsize=FS["axis_label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.set_title(f"{ca_label} {bin_um}µm — clusters",
                 fontsize=FS["title"], fontweight="bold")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=col, markersize=8, label=lbl)
               for lbl, col in colour_map.items()]
    n_clusters = len(colour_map)
    ncol = 2 if n_clusters > 16 else 1
    fig_w = 5 + 1.2 * ncol
    fig.set_size_inches(fig_w, 5)
    ax.legend(handles=handles, fontsize=FS["legend"], ncol=ncol,
              loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False,
              borderpad=0, handletextpad=0.3, columnspacing=0.5, labelspacing=0.3)
    fig.tight_layout()
    save_fig_rasterised(fig, os.path.join(fig_dir,
        f"{_ca_slug(ca_label)}_{bin_um:02d}µm_UMAP_clusters"), dpi_png=600)

def plot_spatial_clusters(adata, bin_um, ca_label, fig_dir):
    """
    Full-tissue spatial cluster map.
    For 2um: uses kNN-projected labels (all bins covered).
    For 4/8um: uses direct Leiden labels.
    """
    x, y       = _spatial_coords(adata)
    labels     = adata.obs["leiden_projected"] \
                 if "leiden_projected" in adata.obs.columns \
                 else adata.obs["leiden"]
    colour_map = _batlow_colours(labels)

    dot_size = DOT_SIZE.get(bin_um, 1.0)
    n_clusters = len(colour_map)
    ncol       = 2 if n_clusters > 16 else 1
    fig, ax = plt.subplots(figsize=(6, 6))
    for lbl, col in colour_map.items():
        mask = (labels == lbl).values
        ax.scatter(x[mask], y[mask],
                   c=[col], s=dot_size, alpha=1.0, label=lbl,
                   marker=".", linewidths=0, edgecolors="none",
                   rasterized=True)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"{ca_label} {bin_um}µm — spatial clusters",
                 fontsize=FS["title"], fontweight="bold")
    ax.set_xlabel("x (µm)", fontsize=FS["axis_label"])
    ax.set_ylabel("y (µm)", fontsize=FS["axis_label"])
    ax.tick_params(labelsize=FS["tick"])
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=col, markersize=8, label=lbl)
               for lbl, col in colour_map.items()]
    ax.legend(handles=handles, fontsize=FS["legend"], ncol=ncol,
              loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False,
              borderpad=0, handletextpad=0.3, columnspacing=0.5, labelspacing=0.3)
    fig.tight_layout()
    save_fig_rasterised(fig, os.path.join(fig_dir,
        f"{_ca_slug(ca_label)}_{bin_um:02d}µm_Spatial_clusters"), dpi_png=600)

# =============================================================================
# kNN LABEL PROJECTION — chunked, memory-safe
# Projects cluster labels from subsample back to all bins.
# Same principle as Seurat MapQuery — find nearest neighbour in PCA space,
# assign that neighbour's cluster label.
# Processes full dataset in PROJ_CHUNK-bin chunks to avoid one giant dense
# matrix (~600 MB peak per chunk instead of ~48 GB all at once).
# =============================================================================
def project_labels_to_full(adata_sub, adata_full, hvg_genes):
    from pynndescent import NNDescent

    print("    Projecting cluster labels to full dataset via kNN (chunked) ...")
    t0 = time.time()

    # Subset full data to same HVGs and normalise
    common  = [g for g in hvg_genes if g in adata_full.var_names]
    adata_f = adata_full[:, common].copy()
    sc.pp.normalize_total(adata_f)
    sc.pp.log1p(adata_f)
    sc.pp.scale(adata_f, max_value=10)

    pca_components = adata_sub.varm["PCs"]           # genes x n_pcs
    pca_mean       = adata_sub.uns["pca"].get("mean", None)
    X_sub_pca      = adata_sub.obsm["X_pca"]

    # Build kNN index on subsample first (done once)
    print(f"      Building kNN index on {X_sub_pca.shape[0]:,} subsample bins ...")
    index = NNDescent(X_sub_pca, n_neighbors=15, random_state=RANDOM_SEED)
    index.prepare()

    # Project full data in chunks to cap peak RAM
    n_full    = adata_f.n_obs
    projected = np.empty(n_full, dtype=object)

    for start in range(0, n_full, PROJ_CHUNK):
        end    = min(start + PROJ_CHUNK, n_full)
        chunk  = adata_f.X[start:end]
        X_chunk = chunk.toarray() if hasattr(chunk, "toarray") else np.array(chunk)
        if pca_mean is not None:
            X_chunk = X_chunk - pca_mean
        X_chunk_pca        = X_chunk @ pca_components
        neighbours, _      = index.query(X_chunk_pca, k=1)
        projected[start:end] = adata_sub.obs["leiden"].values[neighbours[:, 0]]
        del X_chunk, X_chunk_pca
        gc.collect()
        if start % 500_000 == 0 and start > 0:
            print(f"      ... {start:,} / {n_full:,} bins projected")

    del adata_f
    gc.collect()

    adata_full.obs["leiden_projected"] = projected.astype(str)
    print(f"    Label projection done ({time.time()-t0:.1f}s) — "
          f"{len(np.unique(projected))} clusters across {adata_full.n_obs:,} bins")
    return adata_full

# =============================================================================
# PROCESSING PIPELINE — one CA at a time
# =============================================================================
def process_capture_area(ca_label, base_dir, slice_name):
    print(f"\n{'='*60}")
    print(f"  {ca_label}  |  PID {os.getpid()}  |  {_THREADS_PER_CA}/{_N_CORES} CPU threads")
    print(f"{'='*60}")

    fig_dir = os.path.join(base_dir, "figures_visiumHD_2_4_8um")
    os.makedirs(fig_dir, exist_ok=True)

    ca_t0 = time.time()

    for bin_um in BIN_SIZES:
        print(f"\n  [{ca_label}] Loading {bin_um}µm ...")
        t0 = time.time()
        try:
            adata_raw = load_visiumhd_bin(base_dir, bin_um, slice_name)
        except Exception as e:
            print(f"    WARNING: skipping {bin_um}µm — {e}")
            continue

        # == QC plots on full dataset =======================================
        plot_qc_violin(adata_raw, bin_um, ca_label, fig_dir)
        plot_spatial_ncount(adata_raw, bin_um, ca_label, fig_dir)

        # == Determine if subsampling needed; keep full copy if so ==========
        subsample_n      = SUBSAMPLE.get(bin_um)
        needs_projection = subsample_n is not None and adata_raw.n_obs > subsample_n
        adata_full       = adata_raw.copy() if needs_projection else None

        adata = adata_raw.copy()
        del adata_raw
        gc.collect()

        # == Subsample for clustering if needed =============================
        if needs_projection:
            print(f"    Subsampling {bin_um}µm: {adata.n_obs:,} -> {subsample_n:,} "
                  f"bins for clustering (labels projected back to all bins)")
            sc.pp.subsample(adata, n_obs=subsample_n, random_state=RANDOM_SEED)

        # == Normalise → HVG → scale → PCA ==================================
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat")
        hvg_genes = adata.var_names[adata.var["highly_variable"]].tolist()
        adata     = adata[:, adata.var["highly_variable"]].copy()
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=N_PCS, random_state=RANDOM_SEED,
                  svd_solver="arpack")   # arpack uses all OMP threads
        plot_elbow(adata, bin_um, ca_label, fig_dir)

        # == Neighbours → Leiden → UMAP ======================================
        sc.pp.neighbors(adata, n_pcs=N_PCS, use_rep="X_pca",
                        n_neighbors=20, metric="cosine",
                        random_state=RANDOM_SEED)
        sc.tl.leiden(adata, resolution=CLUSTER_RES, random_state=RANDOM_SEED,
                     key_added="leiden")
        print(f"    {adata.obs['leiden'].nunique()} clusters at res={CLUSTER_RES}")
        sc.tl.umap(adata, n_components=2, min_dist=0.3,
                   spread=1.0, random_state=RANDOM_SEED)
        plot_umap_clusters(adata, bin_um, ca_label, fig_dir)

        # == Spatial cluster map =============================================
        if needs_projection and adata_full is not None:
            adata_full = project_labels_to_full(adata, adata_full, hvg_genes)
            plot_spatial_clusters(adata_full, bin_um, ca_label, fig_dir)
            del adata_full
        else:
            plot_spatial_clusters(adata, bin_um, ca_label, fig_dir)

        elapsed = time.time() - t0
        print(f"    [{ca_label}] {bin_um}µm done in {elapsed/60:.1f} min")

        del adata
        gc.collect()

    total = time.time() - ca_t0
    print(f"\n  {ca_label} COMPLETE in {total/60:.1f} min. Figures -> {fig_dir}")

# =============================================================================
# MAIN — sequential CAs, all cores per CA
# =============================================================================
if __name__ == "__main__":
    print(f"\nCPU cores detected: {_N_CORES}")
    print(f"Running {len(CA_DIRS)} CAs sequentially, {_THREADS_PER_CA} threads each")
    print(f"Projection chunk size: {PROJ_CHUNK:,} bins\n")

    t_global = time.time()

    for ca_label, base_dir in CA_DIRS.items():
        process_capture_area(ca_label, base_dir, SLICE_NAMES[ca_label])
        print(f"[OK] {ca_label} finished\n")

    total_min = (time.time() - t_global) / 60
    print(f"\nAll capture areas complete in {total_min:.1f} min.")
