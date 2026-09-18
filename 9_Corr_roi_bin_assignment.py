"""
9_Corr_roi_bin_assignment.py

Places the 50 um nanoindentation grid onto the spatial transcriptome and
assigns 4 um bins to each measurement. Produces the table that every
downstream correlation script reads.

The ROI boundary drawn in SpatialX is the nanoindentation grid boundary, so
rather than laying a fixed grid from an origin, the script projects the ROI
bins onto the rotation axes, fits the grid to the observed bin cloud, and
divides that extent into n_x x n_y equal cells. Every ROI bin is therefore
assigned to exactly one cell, the grid fits regardless of tissue shrinkage,
and corner annotations are needed only to fix the rotation.

Parameters
    N_BINS_PER_BOX    100, bins retained per cell, nearest to its centroid.
                      Taking the nearest rather than all excludes cell edges,
                      where assignment is most sensitive to residual
                      registration error
    MIN_BINS_PER_BOX  20, cells below this are flagged low-confidence and kept
    VISCOSITY_MAX     1.0 N s m^-1, cap for Maxwell fit failures
    GRID_ORIGIN       1, the pipeline emits 1-based indices. The geometry
                      solver works 0-based and converts once at load

Actions required
    In SpatialX, annotate one 4 um bin at each accessible ROI corner under the
    ROI_boundary column:
        <ROI>_BotL   bottom-left    <ROI>_BotR   bottom-right
        <ROI>_TopL   top-left       <ROI>_TopR   top-right
    Where a corner falls in missing tissue, annotate a bin along each adjacent
    edge instead; such a bin has one grid coordinate known and gives axis
    direction without scale. Three annotations per ROI give a robust solution.
    Map each label to its grid position in corner_annotations, then set
    METADATA_FILE and NI_ROOT.
        py 9_Corr_roi_bin_assignment.py

Input
    <METADATA_FILE>     SpatialX export, one row per 4 um bin, with cluster
                        assignments, tissue_compartment, coordinates, ROIs and
                        ROI_boundary annotations
    <NI_ROOT>/          the per-slice all-samples CSVs from script 1
Output
    metadata_with_nano.tsv      one row per 4 um bin with its matched
                                measurement, ROI, compartment and mechanical
                                parameters. Input to scripts 5, 12, 15 and 16
    verification_<ROI>.png      recovered grid overlaid on the ROI bins, one
                                per ROI

Note
    Each measurement is keyed on tissue slice, ROI and grid coordinate
    together. Grid coordinates repeat across ROIs and slices, so grouping on
    the coordinate alone merges unrelated measurements. Every bin assigned to
    a measurement inherits the same mechanical value, so constancy of that
    value across the bins confirms the key is correct.

Dependencies
    pip install pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import sys
import re as _re
from datetime import datetime

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Change metadata file name
METADATA_FILE = Path(r"Z:\Visium_Data\001_SpatialX\20260803_FinalAnnotatedMetadata\HTO_Combined_4um_Images\metadata\HTO_Combined_4um_Images_annotated_data.tsv")
METADATA_SEP  = "\t"

# Auto-derived — do not change this line
OUTPUT_DIR    = METADATA_FILE.parent.parent.parent
OUTPUT_CSV    = OUTPUT_DIR / "metadata_with_nano.tsv"

# NI CSVs stay in 001_SpatialX — use absolute paths or relative to script
NI_ROOT       = Path(r"Z:\Visium_Data\001_SpatialX")

N_BINS_PER_BOX   = 140       # bins selected per grid cell (nearest to centroid)
MIN_BINS_PER_BOX = 20        # cells below this are flagged low-confidence
MAX_CENTROID_DIST_FRAC = 0.75
VISCOSITY_MAX    = 1.0       # Ns/m cap for Maxwell fit failures

# Grid indices are 1-based throughout the pipeline: the first indentation is
# X1 Y1, never X0 Y0. Corner annotations below use that convention. The
# geometry solver works in 0-based cell coordinates, so annotations are
# converted once at load time by to_internal_corners().
GRID_ORIGIN      = 1

COL_ROI          = "ROIs"
COL_BOUNDARY     = "ROI_boundary"
COL_X            = "x"
COL_Y            = "y"

# Column in SpatialX metadata that corresponds to "Unique Barcodes"
UNIQUE_BARCODE_COL = "Unique Barcodes"

# =============================================================================
# ROI CONFIGURATION
# =============================================================================

ROI_CONFIGS = {

    "Y1-1": {
        "roi_label"  : "Y1-1",
        "n_x"        : 7,
        "n_y"        : 12,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "young",
        "source_slice": "20240920_13w-01b-ms01-young-Y1-1",
        "corner_annotations": {
            "Y1-1_BotL"    : {"nano_x": 1,    "nano_y": 1   },
            "Y1-1_BotR"    : {"nano_x": 7,   "nano_y": 1   },
            "Y1-1_TopL"    : {"nano_x": 1,    "nano_y": 12  },
            "Y1-1_TopR"    : {"nano_x": 7,   "nano_y": 12  },
            "Y1-1_Top-bord": {"nano_x": None, "nano_y": 12  },
            "Y1-1_Bot-bord": {"nano_x": None, "nano_y": 1   },
            "Y1-1_L-bord"  : {"nano_x": 1,    "nano_y": None},
            "Y1-1_R-bord"  : {"nano_x": 7,   "nano_y": None},
        },
    },
    "Y1-2": {
        "roi_label"  : "Y1-2",
        "n_x"        : 30,
        "n_y"        : 9,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "young",
        "source_slice": "20240920_13w-01b-ms01-young-Y1-1",
        "corner_annotations": {
            "Y1-2_BotL"    : {"nano_x": 1,    "nano_y": 1   },
            "Y1-2_BotR"    : {"nano_x": 30,  "nano_y": 1   },
            "Y1-2_TopL"    : {"nano_x": 1,    "nano_y": 9   },
            "Y1-2_TopR"    : {"nano_x": 30,  "nano_y": 9   },
            "Y1-2_Top-bord": {"nano_x": None, "nano_y": 9   },
            "Y1-2_Bot-bord": {"nano_x": None, "nano_y": 1   },
            "Y1-2_L-bord"  : {"nano_x": 1,    "nano_y": None},
            "Y1-2_R-bord"  : {"nano_x": 30,  "nano_y": None},
        },
    },
    "Y1-3": {
        "roi_label"  : "Y1-3",
        "n_x"        : 19,
        "n_y"        : 4,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "young",
        "source_slice": "20240920_13w-01b-ms01-young-Y1-1",
        "corner_annotations": {
            "Y1-3_BotL"    : {"nano_x": 1,    "nano_y": 1   },
            "Y1-3_BotR"    : {"nano_x": 19,  "nano_y": 1   },
            "Y1-3_TopL"    : {"nano_x": 1,    "nano_y": 4   },
            "Y1-3_TopR"    : {"nano_x": 19,  "nano_y": 4   },
            "Y1-3_Top-bord": {"nano_x": None, "nano_y": 4   },
            "Y1-3_Bot-bord": {"nano_x": None, "nano_y": 1   },
            "Y1-3_L-bord"  : {"nano_x": 1,    "nano_y": None},
            "Y1-3_R-bord"  : {"nano_x": 19,  "nano_y": None},
        },
    },
    "Y1-4": {
        "roi_label"  : "Y1-4",
        "n_x"        : 11,
        "n_y"        : 7,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "young",
        "source_slice": "20240920_13w-01-ms02-young-Y1-2",
        "corner_annotations": {
            "Y1-4_BotL"    : {"nano_x": 1,    "nano_y": 1   },
            "Y1-4_BotR"    : {"nano_x": 11,  "nano_y": 1   },
            "Y1-4_TopL"    : {"nano_x": 1,    "nano_y": 7   },
            "Y1-4_TopR"    : {"nano_x": 11,  "nano_y": 7   },
            "Y1-4_Top-bord": {"nano_x": None, "nano_y": 7   },
            "Y1-4_Bot-bord": {"nano_x": None, "nano_y": 1   },
            "Y1-4_L-bord"  : {"nano_x": 1,    "nano_y": None},
            "Y1-4_R-bord"  : {"nano_x": 11,  "nano_y": None},
        },
    },
    "Y2-1": {
        "roi_label"  : "Y2-1",
        "n_x"        : 7,
        "n_y"        : 13,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "young",
        "source_slice": "20240920_13w-02-ms01-young-Y2",
        "corner_annotations": {
            "Y2-1_BotL"  : {"nano_x": 1   , "nano_y": 1   },
            "Y2-1_BotR"  : {"nano_x": 7   , "nano_y": 1   },
            "Y2-1_TopR"  : {"nano_x": 7   , "nano_y": 13  },
            "Y2-1_L-bord": {"nano_x": 1   , "nano_y": None},
        },
    },
    "Y2-2": {
        "roi_label"  : "Y2-2",
        "n_x"        : 7,
        "n_y"        : 14,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "young",
        "source_slice": "20240920_13w-02-ms01-young-Y2",
        "corner_annotations": {
            "Y2-2_BotL": {"nano_x": 1   , "nano_y": 1   },
            "Y2-2_BotR": {"nano_x": 7   , "nano_y": 1   },
            "Y2-2_TopL": {"nano_x": 1   , "nano_y": 14  },
            "Y2-2_TopR": {"nano_x": 7   , "nano_y": 14  },
        },
    },
    "Y4-1": {
        "roi_label"  : "Y4-1",
        "n_x"        : 6,
        "n_y"        : 19,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "young",
        "source_slice": "20240916_13w-02b-ms01-young-Y4-1",
        "corner_annotations": {
            "Y4-1_BotL": {"nano_x": 1   , "nano_y": 1   },
            "Y4-1_BotR": {"nano_x": 6   , "nano_y": 1   },
            "Y4-1_TopL": {"nano_x": 1   , "nano_y": 19  },
            "Y4-1_TopR": {"nano_x": 6   , "nano_y": 19  },
        },
    },
    "Y4-2": {
        "roi_label"  : "Y4-2",
        "n_x"        : 26,
        "n_y"        : 17,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "young",
        "source_slice": "20240916_13w-02c-ms01-young-Y4-2",
        "corner_annotations": {
            "Y4-2_BotL": {"nano_x": 1   , "nano_y": 1   },
            "Y4-2_BotR": {"nano_x": 26  , "nano_y": 1   },
            "Y4-2_TopL": {"nano_x": 1   , "nano_y": 17  },
            "Y4-2_TopR": {"nano_x": 26  , "nano_y": 17  },
        },
    },
    "Y5-1": {
        "roi_label"  : "Y5-1",
        "n_x"        : 10,
        "n_y"        : 12,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "young",
        "source_slice": "20240917_13w-01-ms01-young-Y5",
        "corner_annotations": {
            "Y5-1_BotL"    : {"nano_x": 1   , "nano_y": 1   },
            "Y5-1_TopL"    : {"nano_x": 1   , "nano_y": 12  },
            "Y5-1_TopR"    : {"nano_x": 10  , "nano_y": 12  },
            "Y5-1_R-bord"  : {"nano_x": 10  , "nano_y": None},
            "Y5-1_Bot-bord": {"nano_x": None, "nano_y": 1   },
        },
    },
    "Y5-2": {
        "roi_label"  : "Y5-2",
        "n_x"        : 23,
        "n_y"        : 15,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "young",
        "source_slice": "20240917_13w-01-ms01-young-Y5",
        "corner_annotations": {
            "Y5-2_BotR"    : {"nano_x": 23  , "nano_y": 1   },
            "Y5-2_TopR"    : {"nano_x": 23  , "nano_y": 15  },
            "Y5-2_TopL"    : {"nano_x": 1   , "nano_y": 15  },
            "Y5-2_L-bord"  : {"nano_x": 1   , "nano_y": None},
            "Y5-2_Bot-bord": {"nano_x": None, "nano_y": 1   },
        },
    },
    "Y5-3": {
        "roi_label"  : "Y5-3",
        "n_x"        : 23,
        "n_y"        : 4,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "young",
        "source_slice": "20240917_13w-01-ms01-young-Y5",
        "corner_annotations": {
            "Y5-3_BotR"    : {"nano_x": 23  , "nano_y": 1   },
            "Y5-3_TopR"    : {"nano_x": 23  , "nano_y": 4   },
            "Y5-3_Top-bord": {"nano_x": None, "nano_y": 4   },
            "Y5-3_Bot-bord": {"nano_x": None, "nano_y": 1   },
        },
    },
    "A1-1": {
        "roi_label"  : "A1-1",
        "n_x"        : 10,
        "n_y"        : 21,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "aged",
        "source_slice": "20240909_14m-02-ms02-aged-A1-1",
        "corner_annotations": {
            "A1-1_BotL": {"nano_x": 1   , "nano_y": 1   },
            "A1-1_BotR": {"nano_x": 10  , "nano_y": 1   },
            "A1-1_TopL": {"nano_x": 1   , "nano_y": 21  },
            "A1-1_TopR": {"nano_x": 10  , "nano_y": 21  },
        },
    },
    "A1-2": {
        "roi_label"  : "A1-2",
        "n_x"        : 8,
        "n_y"        : 21,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "aged",
        "source_slice": "20240909_14m-02c-ms02-aged-A1-2",
        "corner_annotations": {
            "A1-2_BotL"    : {"nano_x": 1   , "nano_y": 1   },
            "A1-2_BotR"    : {"nano_x": 8   , "nano_y": 1   },
            "A1-2_TopL"    : {"nano_x": 1   , "nano_y": 21  },
            "A1-2_Top-bord": {"nano_x": None, "nano_y": 21  },
            "A1-2_R-bord"  : {"nano_x": 8   , "nano_y": None},
        },
    },
    "A2-1": {
        "roi_label"  : "A2-1",
        "n_x"        : 4,
        "n_y"        : 17,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "aged",
        "source_slice": "20240909_14m-01-ms01-aged-A2-1",
        "corner_annotations": {
            "A2-1_BotL"  : {"nano_x": 1   , "nano_y": 1   },
            "A2-1_BotR"  : {"nano_x": 4   , "nano_y": 1   },
            "A2-1_L-bord": {"nano_x": 1   , "nano_y": None},
            "A2-1_R-bord": {"nano_x": 4   , "nano_y": None},
        },
    },
    "A2-3": {
        "roi_label"  : "A2-3",
        "n_x"        : 4,
        "n_y"        : 14,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "aged",
        "source_slice": "20240909_14m-01-ms01-aged-A2-1",
        "corner_annotations": {
            "A2-3_BotL"    : {"nano_x": 1   , "nano_y": 1   },
            "A2-3_L-bord"  : {"nano_x": 1   , "nano_y": None},
            "A2-3_Bot-bord": {"nano_x": None, "nano_y": 1   },
        },
    },
    "A2-2": {
        "roi_label"  : "A2-2",
        "n_x"        : 10,
        "n_y"        : 20,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "aged",
        "source_slice": "20240909_14m-01b-ms01-aged-A2-2",
        "corner_annotations": {
            "A2-2_BotL": {"nano_x": 1   , "nano_y": 1   },
            "A2-2_BotR": {"nano_x": 10  , "nano_y": 1   },
            "A2-2_TopL": {"nano_x": 1   , "nano_y": 20  },
            "A2-2_TopR": {"nano_x": 10  , "nano_y": 20  },
        },
    },
    "A3-1": {
        "roi_label"  : "A3-1",
        "n_x"        : 11,
        "n_y"        : 8,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "aged",
        "source_slice": "20240906_14m-01b-ms01-aged-A3",
        "corner_annotations": {
            "A3-1_BotL": {"nano_x": 1   , "nano_y": 1   },
            "A3-1_BotR": {"nano_x": 11  , "nano_y": 1   },
            "A3-1_TopL": {"nano_x": 1   , "nano_y": 8   },
            "A3-1_TopR": {"nano_x": 11  , "nano_y": 8   },
        },
    },
    "A3-2": {
        "roi_label"  : "A3-2",
        "n_x"        : 29,
        "n_y"        : 4,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "aged",
        "source_slice": "20240906_14m-01b-ms01-aged-A3",
        "corner_annotations": {
            "A3-2_BotL"    : {"nano_x": 1   , "nano_y": 1   },
            "A3-2_TopL"    : {"nano_x": 1   , "nano_y": 4   },
            "A3-2_Top-bord": {"nano_x": None, "nano_y": 4   },
            "A3-2_Bot-bord": {"nano_x": None, "nano_y": 1   },
        },
    },
    "A3-3": {
        "roi_label"  : "A3-3",
        "n_x"        : 15,
        "n_y"        : 7,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "aged",
        "source_slice": "20240906_14m-01b-ms01-aged-A3",
        "corner_annotations": {
            "A3-3_BotL"    : {"nano_x": 1   , "nano_y": 1   },
            "A3-3_TopL"    : {"nano_x": 1   , "nano_y": 7   },
            "A3-3_Top-bord": {"nano_x": None, "nano_y": 7   },
            "A3-3_Bot-bord": {"nano_x": None, "nano_y": 1   },
        },
    },
    "A6-1": {
        "roi_label"  : "A6-1",
        "n_x"        : 5,
        "n_y"        : 13,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "aged",
        "source_slice": "20240903_14m-02-ms02-aged-A6",
        "corner_annotations": {
            "A6-1_BotL": {"nano_x": 1   , "nano_y": 1   },
            "A6-1_BotR": {"nano_x": 5   , "nano_y": 1   },
            "A6-1_TopL": {"nano_x": 1   , "nano_y": 13  },
            "A6-1_TopR": {"nano_x": 5   , "nano_y": 13  },
        },
    },
    "A6-2": {
        "roi_label"  : "A6-2",
        "n_x"        : 10,
        "n_y"        : 22,
        "nano_csv"   : None,          # resolved by find_nano_csv()
        "sample_id"  : None,
        "age_group"  : "aged",
        "source_slice": "20240903_14m-02-ms02-aged-A6",
        "corner_annotations": {
            "A6-2_BotL": {"nano_x": 1   , "nano_y": 1   },
            "A6-2_BotR": {"nano_x": 10  , "nano_y": 1   },
            "A6-2_TopL": {"nano_x": 1   , "nano_y": 22  },
            "A6-2_TopR": {"nano_x": 10  , "nano_y": 22  },
        },
    },
}


NANO_SCALAR_COLS = {
    "Elasticity_l_Pa"   : "Elasticity_l (Pa)",
    "Elasticity_eff_Pa" : "Elasticity_eff (Pa)",
    "Loss"              : "Loss",
    "Energy_l_fJ"       : "Energy_l (fJ)",
    "Energy_ad_fJ"      : "Energy_ad (fJ)",
    "Viscosity_Ns_m"    : "Viscosity (Ns/m)",
    "Power_l"           : "Power_l",
    "Z_um"              : "Z (um)",
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def find_nano_csv(roi_key, cfg):
    """Locate the ROI CSV for this ROI inside NI_ROOT.

    Files written by 7_NI_roi_crop.py are named

        NI_<slice>_all_samples_<ROI>.csv

    so the slice prefix varies. Matching on the trailing _<ROI>.csv is enough
    to identify one uniquely. An explicit "nano_csv" in the config wins if set.
    Older NI_<ROI>_all_samples.csv files are accepted as a fallback.
    """
    explicit = cfg.get('nano_csv')
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        print(f"    configured nano_csv not found: {p}")

    hits = sorted(NI_ROOT.glob(f"NI_*_all_samples_{roi_key}.csv"))
    if cfg.get('source_slice'):
        exact = NI_ROOT / f"NI_{cfg['source_slice']}_all_samples_{roi_key}.csv"
        if exact.is_file():
            return exact
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise FileNotFoundError(
            f"{roi_key}: {len(hits)} files match NI_*_all_samples_{roi_key}.csv "
            f"in {NI_ROOT}. Set 'nano_csv' explicitly for this ROI. "
            f"Found: {[h.name for h in hits]}")

    legacy = NI_ROOT / f"NI_{roi_key}_all_samples.csv"
    if legacy.is_file():
        print(f"    using legacy filename {legacy.name}")
        return legacy

    raise FileNotFoundError(
        f"{roi_key}: no NI_*_all_samples_{roi_key}.csv in {NI_ROOT}")


def to_internal_corners(corner_annotations):
    """Config annotations are 1-based; the solver expects 0-based cells."""
    out = {}
    for label, gp in corner_annotations.items():
        out[label] = {
            'nano_x': None if gp['nano_x'] is None else int(gp['nano_x']) - GRID_ORIGIN,
            'nano_y': None if gp['nano_y'] is None else int(gp['nano_y']) - GRID_ORIGIN,
        }
    return out


def load_nano_csv(path, sample_id=None):
    df = pd.read_csv(path)
    if sample_id is not None:
        df = df[df['Sample'] == sample_id].copy()
    df = df[df['IndentationPresent'] == True].dropna(subset=['X', 'Y'])
    df['X_orig'] = df['X'].astype(int)
    df['Y_orig'] = df['Y'].astype(int)

    # Anchor on the known 1-based origin rather than the observed minimum.
    # Using min() would silently shift the whole grid whenever the first row
    # or column was removed upstream by the contact / fit QC.
    df['X_idx'] = df['X_orig'] - GRID_ORIGIN
    df['Y_idx'] = df['Y_orig'] - GRID_ORIGIN
    lost = df[(df['X_idx'] < 0) | (df['Y_idx'] < 0)]
    if len(lost):
        print(f"    WARNING: {len(lost)} rows have X or Y below {GRID_ORIGIN} "
              f"and were dropped. Was this CSV renumbered by 7_NI_roi_crop.py?")
        df = df[(df['X_idx'] >= 0) & (df['Y_idx'] >= 0)]

    if 'Viscosity (Ns/m)' in df.columns:
        df.loc[df['Viscosity (Ns/m)'].abs() > VISCOSITY_MAX,
               'Viscosity (Ns/m)'] = np.nan
        
    return df.set_index(['X_idx', 'Y_idx'])


def recover_axes_from_corners(meta, corner_annotations, roi_bins_xy, n_x, n_y):
    """
    Recover x_unit and y_unit rotation axes from annotated corner bins.
    Also returns the extrapolated X0Y0 origin (for the verification plot
    gold star only -- not used for bin assignment).

    Parameters
    ----------
    meta               : full metadata DataFrame
    corner_annotations : dict  label -> {nano_x, nano_y}
    roi_bins_xy        : (n, 2) array of ROI bin positions
    n_x, n_y           : grid dimensions

    Returns
    -------
    x_unit          : unit vector along X axis
    y_unit          : unit vector along Y axis
    origin          : extrapolated X0Y0 position (for plotting only)
    corner_positions: dict  label -> (x, y) centroid of annotated bins
    """
    corner_pos = {}
    for label, gp in corner_annotations.items():
        mask = meta[COL_BOUNDARY].astype(str) == label
        if mask.sum() == 0:
            # Border annotations are optional — skip if not annotated in SpatialX.
            # Full corners are also skipped here; an error is raised later if
            # the remaining annotations are insufficient to recover the axes.
            print(f"    '{label}': not found in metadata — skipping")
            continue
        corner_pos[label] = np.array([
            meta.loc[mask, COL_X].mean(),
            meta.loc[mask, COL_Y].mean(),
        ])
        nx_str = str(gp['nano_x']) if gp['nano_x'] is not None else '?'
        ny_str = str(gp['nano_y']) if gp['nano_y'] is not None else '?'
        print(f"    '{label}': {mask.sum()} bin(s), "
              f"pos = ({corner_pos[label][0]:.1f}, {corner_pos[label][1]:.1f}), "
              f"grid = X{nx_str} Y{ny_str}")
 
    # Separate full corners (both coords known) from border annotations
    # (one coord is None: Top/Bot-bord have nano_x=None, L/R-bord have nano_y=None)
    full_corners = {lbl: pos for lbl, pos in corner_pos.items()
                    if corner_annotations[lbl]['nano_x'] is not None
                    and corner_annotations[lbl]['nano_y'] is not None}
    n_total  = len(corner_pos)
    n_full   = len(full_corners)
    n_borders = n_total - n_full
 
    if n_total == 0:
        raise ValueError("No corner or border annotations found for this ROI.")
    if n_borders > 0:
        print(f"    Border annotations detected: {n_borders} "
              f"(Top/Bot/L/R-bord — one coordinate unknown)")
    if n_total == 2:
        print("    WARNING: only 2 annotations. Rotation may be less accurate.")

    # ---------------------------------------------------------------
    # SINGLE-ANNOTATION FALLBACK: PCA on the full ROI bin cloud.
    # Triggered when only 1 total annotation exists (corner or border).
    # PCA finds the principal axes of the bin cloud shape. The longer
    # PC maps to the longer grid dimension. The single annotation
    # anchors the sign via a bin-count test across all 4 sign combos.
    # Reliable for aspect ratio >= 1.5:1; less so for near-square grids.
    # ---------------------------------------------------------------
    if n_total == 1:
        lbl0 = list(corner_pos.keys())[0]
        cpos = corner_pos[lbl0]
        cnx  = corner_annotations[lbl0]['nano_x']
        cny  = corner_annotations[lbl0]['nano_y']
        # Use 0 for any unknown coordinate (border point) in the sign test
        anc_nx = cnx if cnx is not None else 0
        anc_ny = cny if cny is not None else 0
 
        aspect = max(n_x, n_y) / min(n_x, n_y)
        print(f"    WARNING: only 1 annotation. "
              f"Using PCA fallback (aspect ratio {aspect:.2f}:1).")
        if aspect < 1.5:
            print("    WARNING: aspect ratio < 1.5 — PCA may be unreliable.")
 
        cov = np.cov(roi_bins_xy.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        pc1 = eigenvectors[:, idx[0]]
        pc2 = eigenvectors[:, idx[1]]
        x_cand, y_cand = (pc1, pc2) if n_x >= n_y else (pc2, pc1)
 
        best_count, x_unit, y_unit = -1, None, None
        for sx in [+1, -1]:
            for sy in [+1, -1]:
                xu   = sx * x_cand
                yu   = sy * y_cand
                orig = cpos - anc_nx * 50.0 * xu - anc_ny * 50.0 * yu
                sh   = roi_bins_xy - orig
                px   = (sh @ xu) / 50.0
                py   = (sh @ yu) / 50.0
                count = int(((px >= -0.5) & (px <= n_x - 0.5) &
                              (py >= -0.5) & (py <= n_y - 0.5)).sum())
                if count > best_count:
                    best_count, x_unit, y_unit = count, xu, yu
        print(f"    PCA: {best_count:,}/{len(roi_bins_xy):,} bins within grid bounds")
 
    else:
        # ---------------------------------------------------------------
        # MULTI-ANNOTATION PATH
        # Handles all combinations of full corners and border annotations.
        #
        # Full corners (both nano_x and nano_y known):
        #   Same-row pair  (same nano_y, both nano_x known) → x_unit (scale-based)
        #   Same-col pair  (same nano_x, both nano_y known) → y_unit (scale-based)
        #
        # Border annotations (one coord is None):
        #   Top/Bot-bord: nano_y is fixed (0 or n_y-1), nano_x is unknown
        #   L/R-bord:     nano_x is fixed (0 or n_x-1), nano_y is unknown
        #   Paired with a full corner on the SAME edge → axis direction (scale-free)
        #   Sign is determined from the known anchor coordinate via _border_sign().
        #
        # Fallback hierarchy when one or both axes are still unknown:
        #   1. Derive missing axis as perpendicular + bin-count sign test
        #   2. Diagonal-only full corners → analytic perpendicularity solver
        #      (direction-check only, scale-invariant for shrunken tissue)
        # ---------------------------------------------------------------
        pts = [(corner_pos[lbl],
                corner_annotations[lbl]['nano_x'],
                corner_annotations[lbl]['nano_y'])
               for lbl in corner_pos]
 
        x_vecs = []
        y_vecs = []
 
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                pi, xi_n, yi_n = pts[i]
                pj, xj_n, yj_n = pts[j]
 
                # --- X-AXIS: same horizontal edge (same known nano_y) ---
                if yi_n is not None and yj_n is not None and yi_n == yj_n:
                    raw      = pj - pi
                    raw_norm = np.linalg.norm(raw)
                    if raw_norm < 1e-6:
                        continue
                    if xi_n is not None and xj_n is not None and xi_n != xj_n:
                        # Both fully known: scale-based (most accurate)
                        v = raw / ((xj_n - xi_n) * 50.0)
                        x_vecs.append(v / np.linalg.norm(v))
                    elif xi_n is None or xj_n is None:
                        # One is a border: direction-only, sign from known anchor
                        sign = _border_sign(xi_n, xj_n, n_x)
                        if sign is not None:
                            x_vecs.append(sign * (raw / raw_norm))
                            print(f"    Border pair on horizontal edge "
                                  f"(nano_y={yi_n}): X-axis direction used")
 
                # --- Y-AXIS: same vertical edge (same known nano_x) ---
                if xi_n is not None and xj_n is not None and xi_n == xj_n:
                    raw      = pj - pi
                    raw_norm = np.linalg.norm(raw)
                    if raw_norm < 1e-6:
                        continue
                    if yi_n is not None and yj_n is not None and yi_n != yj_n:
                        # Both fully known: scale-based
                        v = raw / ((yj_n - yi_n) * 50.0)
                        y_vecs.append(v / np.linalg.norm(v))
                    elif yi_n is None or yj_n is None:
                        # One is a border: direction-only
                        sign = _border_sign(yi_n, yj_n, n_y)
                        if sign is not None:
                            y_vecs.append(sign * (raw / raw_norm))
                            print(f"    Border pair on vertical edge "
                                  f"(nano_x={xi_n}): Y-axis direction used")
 
        x_unit = (np.sum(x_vecs, axis=0) / np.linalg.norm(np.sum(x_vecs, axis=0))
                   if x_vecs else None)
        y_unit = (np.sum(y_vecs, axis=0) / np.linalg.norm(np.sum(y_vecs, axis=0))
                   if y_vecs else None)
 
        if x_unit is None and y_unit is not None:
            x_unit = _pick_perpendicular(y_unit, pts, roi_bins_xy, n_x, n_y, 'x')
            print("    X derived as perpendicular to Y")
        elif y_unit is None and x_unit is not None:
            y_unit = _pick_perpendicular(x_unit, pts, roi_bins_xy, n_x, n_y, 'y')
            print("    Y derived as perpendicular to X")
        elif x_unit is None and y_unit is None:
            # No same-edge pairs found — must be a diagonal-only configuration.
            # Analytic perpendicularity solver: direction-check only so it is
            # scale-invariant and handles tissue shrinkage correctly.
            full_pts = [(corner_pos[lbl],
                         corner_annotations[lbl]['nano_x'],
                         corner_annotations[lbl]['nano_y'])
                        for lbl in full_corners]
            diag_pairs = [
                (full_pts[i], full_pts[j])
                for i in range(len(full_pts))
                for j in range(i + 1, len(full_pts))
                if full_pts[i][1] != full_pts[j][1] and full_pts[i][2] != full_pts[j][2]
            ]
            if not diag_pairs:
                raise ValueError(
                    "Cannot determine rotation. Provide at least one same-edge "
                    "annotation pair: same-row corners/borders for X, or "
                    "same-column corners/borders for Y."
                )
            print("    Diagonal pair — using analytic perpendicularity solver "
                  "(direction-check, scale-invariant)")
            (p1, xi_n1, yi_n1), (p2, xi_n2, yi_n2) = diag_pairs[0]
            delta      = p2 - p1
            delta_norm = np.linalg.norm(delta)
            R = (xi_n2 - xi_n1) * 50.0
            S = (yi_n2 - yi_n1) * 50.0
            phi  = np.arctan2(S, R)
            base = np.arctan2(delta[1], delta[0])
 
            best_count, x_unit, y_unit = -1, None, None
            for theta in [base - phi, base + phi]:
                xu = np.array([np.cos(theta), np.sin(theta)])
                for yu in [np.array([-xu[1], xu[0]]),
                            np.array([ xu[1], -xu[0]])]:
                    # Direction check only — ignores magnitude so tissue
                    # shrinkage does not affect the solution.
                    predicted = R * xu + S * yu
                    pred_norm = np.linalg.norm(predicted)
                    if pred_norm < 1e-10 or delta_norm < 1e-10:
                        continue
                    cos_sim = np.dot(predicted, delta) / (pred_norm * delta_norm)
                    if cos_sim < 0.999:   # < ~2.6 degree tolerance
                        continue
                    orig = p1 - xi_n1 * 50.0 * xu - yi_n1 * 50.0 * yu
                    sh   = roi_bins_xy - orig
                    px   = (sh @ xu) / 50.0
                    py   = (sh @ yu) / 50.0
                    count = int(((px >= -0.5) & (px <= n_x - 0.5) &
                                  (py >= -0.5) & (py <= n_y - 0.5)).sum())
                    if count > best_count:
                        best_count, x_unit, y_unit = count, xu, yu
            print(f"    Diagonal solver: {best_count:,}/{len(roi_bins_xy):,} "
                  f"bins within grid bounds")
 
    x_unit, y_unit = _enforce_perpendicular(x_unit, y_unit)
    angle_deg = np.degrees(np.arctan2(x_unit[1], x_unit[0]))
    print(f"    X-axis angle: {angle_deg:.1f} deg")
    print(f"    X-unit: ({x_unit[0]:.4f}, {x_unit[1]:.4f})")
    print(f"    Y-unit: ({y_unit[0]:.4f}, {y_unit[1]:.4f})")
 
    # Extrapolate origin for plotting only — skip border annotations (None coords)
    origins = []
    for lbl, pos in corner_pos.items():
        xi_n = corner_annotations[lbl]['nano_x']
        yi_n = corner_annotations[lbl]['nano_y']
        if xi_n is not None and yi_n is not None:
            origins.append(pos - xi_n * 50.0 * x_unit - yi_n * 50.0 * y_unit)
    if not origins:
        origins = [roi_bins_xy.mean(axis=0)]
    origin = np.mean(origins, axis=0)
    print(f"    X0Y0 origin (reference only): ({origin[0]:.1f}, {origin[1]:.1f})")
 
    def _rect(idx, n):
        if idx == 0:
            return 0.0
        if idx == n - 1:
            return float(n)
        return float(idx) + 0.5

    corner_pts = []
    border_pts = []
    for lbl, pos in corner_pos.items():
        xi_n = corner_annotations[lbl]['nano_x']
        yi_n = corner_annotations[lbl]['nano_y']
        if xi_n is not None and yi_n is not None:
            corner_pts.append((float(pos[0]), float(pos[1]),
                               _rect(int(xi_n), n_x), _rect(int(yi_n), n_y)))
        elif xi_n is not None or yi_n is not None:
            border_pts.append((
                float(pos[0]), float(pos[1]),
                None if xi_n is None else _rect(int(xi_n), n_x),
                None if yi_n is None else _rect(int(yi_n), n_y)))

    return (x_unit, y_unit, origin,
            {lbl: tuple(p) for lbl, p in corner_pos.items()},
            corner_pts, border_pts)
 
 
def _border_sign(i_coord, j_coord, n_max):
    """
    Determine the sign multiplier for (pj - pi) to give the positive
    axis direction, for a same-edge pair where one coordinate is None
    (border annotation: Top/Bot-bord or L/R-bord).
 
    i_coord : nano_x or nano_y of point i (None if border annotation)
    j_coord : nano_x or nano_y of point j (None if border annotation)
    n_max   : n_x or n_y (total steps along this axis)
 
    Returns +1, -1, or None (ambiguous — both are borders).
 
    Logic:
      A corner at coord=0 is the leftmost/bottommost anchor. Any border
      on the same edge is to its right/above, so (p_border - p_anchor)
      is in the +axis direction → sign = +1.
 
      A corner at coord=n_max-1 is the rightmost/topmost anchor. Any
      border on the same edge is to its left/below, so (p_border -
      p_anchor) is in the -axis direction → sign = -1.
    """
    if i_coord is not None and j_coord is None:
        # pi is full corner, pj is border
        if i_coord == 0:
            return +1          # border is to the right/above of left/bottom corner
        if i_coord == n_max - 1:
            return -1          # border is to the left/below of right/top corner
        return None            # middle position, ambiguous
    elif j_coord is not None and i_coord is None:
        # pj is full corner, pi is border
        if j_coord == 0:
            return -1          # corner is left of border → pj-pi is -axis
        if j_coord == n_max - 1:
            return +1          # corner is right of border → pj-pi is +axis
        return None
    return None                # both are borders, cannot determine sign
 
 
def _pick_perpendicular(known_unit, pts, roi_bins_xy, n_x, n_y, axis):
    perp_a = np.array([-known_unit[1],  known_unit[0]])
    perp_b = np.array([ known_unit[1], -known_unit[0]])
    best, best_cand = -1, None
    for cand, name in [(perp_a, 'CCW'), (perp_b, 'CW')]:
        x_cand = known_unit if axis == 'x' else cand
        y_cand = cand       if axis == 'x' else known_unit
        pos0, xi0, yi0 = pts[0]
        orig = pos0 - xi0*50.0*x_cand - yi0*50.0*y_cand
        sh   = roi_bins_xy - orig
        px   = (sh @ x_cand) / 50.0
        py   = (sh @ y_cand) / 50.0
        count = int(((px >= -0.5) & (px <= n_x-0.5) &
                     (py >= -0.5) & (py <= n_y-0.5)).sum())
        if count > best:
            best, best_cand = count, cand
    return best_cand
 
 
def _enforce_perpendicular(x_unit, y_unit):
    x_norm  = x_unit / np.linalg.norm(x_unit)
    angle_x = np.arctan2(x_norm[1], x_norm[0])
    angle_y = np.arctan2(y_unit[1], y_unit[0])
    diff    = (angle_y - angle_x) % (2 * np.pi)
    y_norm  = (np.array([-x_norm[1], x_norm[0]])
               if abs(diff - np.pi/2) < abs(diff - 3*np.pi/2)
               else np.array([x_norm[1], -x_norm[0]]))
    return x_norm, y_norm
 
 
def tissue_of(roi_key):
    return _re.sub(r"-\d+$", "", str(roi_key).strip())


def _pack(ax, ay, origin, corner_pts, source, mode):
    ax = np.asarray(ax, float); ay = np.asarray(ay, float)
    origin = np.asarray(origin, float)
    resid = 0.0
    if corner_pts:
        P = np.array([[c[0], c[1]] for c in corner_pts], float)
        UV = np.array([[c[2], c[3]] for c in corner_pts], float)
        pred = origin + UV[:, [0]] * ax + UV[:, [1]] * ay
        resid = float(np.linalg.norm(P - pred, axis=1).mean())
    sx, sy = float(np.linalg.norm(ax)), float(np.linalg.norm(ay))
    cosang = float(np.dot(ax, ay) / max(sx * sy, 1e-12))
    return dict(ax=ax, ay=ay, origin=origin,
                spacing_x=sx, spacing_y=sy,
                spacing=float((sx + sy) / 2.0),
                angle=float(np.degrees(np.arctan2(ax[1], ax[0]))),
                shear=float(90.0 - np.degrees(np.arccos(np.clip(cosang, -1, 1)))),
                reflected=bool(ax[0] * ay[1] - ax[1] * ay[0] < 0),
                resid=resid, n=len(corner_pts), source=source, mode=mode)


def fit_affine(corner_pts):
    if len(corner_pts) < 3:
        return None
    UV = np.array([[c[2], c[3], 1.0] for c in corner_pts], float)
    P = np.array([[c[0], c[1]] for c in corner_pts], float)
    if np.linalg.matrix_rank(UV) < 3:
        return None
    M, *_ = np.linalg.lstsq(UV, P, rcond=None)
    return _pack(M[0], M[1], M[2], corner_pts, 'own corners', 'affine')


def border_score(fit, border_pts, n_x, n_y):
    if fit is None or not border_pts:
        return 0.0
    A = np.column_stack([fit['ax'], fit['ay']])
    if abs(np.linalg.det(A)) < 1e-9:
        return -1e9
    score = 0.0
    for x, y, u, v in border_pts:
        uv = np.linalg.solve(A, np.array([x, y], float) - fit['origin'])
        if u is not None:
            score += 1.0 if abs(uv[0] - u) <= 1.0 else -1.0
            score += 1.0 if -1.0 <= uv[1] <= n_y + 1.0 else -1.0
        else:
            score += 1.0 if abs(uv[1] - v) <= 1.0 else -1.0
            score += 1.0 if -1.0 <= uv[0] <= n_x + 1.0 else -1.0
    return score


def fit_similarity(corner_pts, border_pts=None, n_x=None, n_y=None):
    if len(corner_pts) < 2:
        return None
    g = np.array([c[2] + 1j * c[3] for c in corner_pts], dtype=complex)
    p = np.array([c[0] + 1j * c[1] for c in corner_pts], dtype=complex)
    cands = []
    for reflected in (False, True):
        gg = np.conj(g) if reflected else g
        gm, pm = gg.mean(), p.mean()
        denom = float((np.abs(gg - gm) ** 2).sum())
        if denom < 1e-12:
            continue
        a = ((p - pm) * np.conj(gg - gm)).sum() / denom
        b = pm - a * gm
        ax = np.array([a.real, a.imag])
        ay = np.array([-a.imag, a.real])
        if reflected:
            ay = -ay
        cands.append(_pack(ax, ay, [b.real, b.imag], corner_pts,
                           'own corners', 'similarity'))
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    tol = 0.25 * max(c['spacing'] for c in cands)
    if abs(cands[0]['resid'] - cands[1]['resid']) < tol and border_pts:
        sc = [border_score(c, border_pts, n_x, n_y) for c in cands]
        if sc[0] != sc[1]:
            pick = cands[int(np.argmax(sc))]
            print(f"           reflection resolved by border annotations "
                  f"(scores {sc[0]:.0f} vs {sc[1]:.0f})")
            return pick
    return min(cands, key=lambda c: c['resid'])


def fit_with_spacing(corner_pts, x_unit, y_unit, spacing,
                     border_pts=None, n_x=None, n_y=None):
    if not corner_pts or spacing is None:
        return None
    P = np.array([[c[0], c[1]] for c in corner_pts], float)
    UV = np.array([[c[2], c[3]] for c in corner_pts], float)
    f = fit_from_annotations(corner_pts, border_pts, n_x, n_y,
                             fallback=spacing)
    if f is not None:
        f['source'] = 'corners + borders'
        return f
    cands = []
    for sgn in (1.0, -1.0):
        ax = spacing * np.asarray(x_unit, float)
        ay = sgn * spacing * np.asarray(y_unit, float)
        origin = (P - UV[:, [0]] * ax - UV[:, [1]] * ay).mean(axis=0)
        cands.append(_pack(ax, ay, origin, corner_pts, 'tissue spacing',
                           'fixed scale'))
    if border_pts:
        sc = [border_score(c, border_pts, n_x, n_y) for c in cands]
        if sc[0] != sc[1]:
            return cands[int(np.argmax(sc))]
    return cands[0]


def _unit(v):
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else np.asarray(v, float) / n


def _axis_from_annotations(corner_pts, border_pts, n_x, n_y, axis):
    ki, oi = (2, 3) if axis == 'x' else (3, 2)
    n_along = n_x if axis == 'x' else n_y
    vecs, dirs = [], []

    full = [c for c in corner_pts]
    for i in range(len(full)):
        for j in range(i + 1, len(full)):
            a, b = full[i], full[j]
            if abs(a[oi] - b[oi]) > 1e-9:
                continue
            d = b[ki] - a[ki]
            if abs(d) < 1e-9:
                continue
            vecs.append((np.array([b[0], b[1]]) - np.array([a[0], a[1]])) / d)

    for bp in (border_pts or []):
        if bp[ki] is not None or bp[oi] is None:
            continue
        for c in full:
            if abs(c[oi] - bp[oi]) > 1e-9:
                continue
            sgn = 1.0 if c[ki] <= 1e-9 else (-1.0 if c[ki] >= n_along - 1e-9
                                             else 0.0)
            if sgn == 0.0:
                continue
            u = _unit(np.array([bp[0], bp[1]]) - np.array([c[0], c[1]]))
            if u is not None:
                dirs.append(sgn * u)

    return (np.mean(vecs, axis=0) if vecs else None,
            _unit(np.mean(dirs, axis=0)) if dirs else None)


def fit_from_annotations(corner_pts, border_pts, n_x, n_y, fallback=None):
    if not corner_pts:
        return None
    ax_v, ax_d = _axis_from_annotations(corner_pts, border_pts, n_x, n_y, 'x')
    ay_v, ay_d = _axis_from_annotations(corner_pts, border_pts, n_x, n_y, 'y')
    if ax_v is None and ay_v is None and ax_d is None and ay_d is None:
        return None

    mag = None
    if ax_v is not None:
        mag = float(np.linalg.norm(ax_v))
    if ay_v is not None:
        mag = float(np.linalg.norm(ay_v)) if mag is None else mag
    if mag is None:
        mag = fallback
    if mag is None:
        return None

    if ax_v is not None:
        ax = ax_v
    elif ax_d is not None:
        ax = ax_d * (float(np.linalg.norm(ay_v)) if ay_v is not None else mag)
    else:
        u = _unit(ay_v if ay_v is not None else np.array([1.0, 0.0]))
        ax = np.array([u[1], -u[0]]) * mag

    if ay_v is not None:
        ay = ay_v
    elif ay_d is not None:
        ay = ay_d * float(np.linalg.norm(ax))
    else:
        u = _unit(ax)
        ay = np.array([-u[1], u[0]]) * float(np.linalg.norm(ax))

    P = np.array([[c[0], c[1]] for c in corner_pts], float)
    UV = np.array([[c[2], c[3]] for c in corner_pts], float)
    origin = (P - UV[:, [0]] * ax - UV[:, [1]] * ay).mean(axis=0)
    src = 'own corners' if (ax_v is not None and ay_v is not None) \
        else 'corners + borders'
    return _pack(ax, ay, origin, corner_pts, src, 'annotation')


def solve_roi(corner_pts, border_pts=None, n_x=None, n_y=None):
    if len(corner_pts) >= 3:
        f = fit_affine(corner_pts)
        if f is not None:
            return f
    f = fit_from_annotations(corner_pts, border_pts, n_x, n_y)
    if f is not None:
        return f
    return fit_similarity(corner_pts, border_pts, n_x, n_y)


def resolve_tissue_spacings(fits):
    by_tissue = {}
    for roi, f in fits.items():
        if f is not None:
            by_tissue.setdefault(tissue_of(roi), []).append(f['spacing'])
    out = {}
    for t, vals in sorted(by_tissue.items()):
        out[t] = float(np.median(vals))
        if len(vals) > 1:
            spread = max(vals) - min(vals)
            flag = '   <- CHECK, ROIs disagree' if spread > 2.0 else ''
            print(f"    {t}: spacing {out[t]:.2f} um from {len(vals)} ROI(s), "
                  f"range {min(vals):.2f} to {max(vals):.2f}{flag}")
        else:
            print(f"    {t}: spacing {out[t]:.2f} um from 1 ROI "
                  f"(no cross-check available)")
    return out


def report_fit(roi, fit):
    if fit is None:
        print(f"  {roi:<8s} NO FIT: too few full corners")
        return
    print(f"  {roi:<8s} cell {fit['spacing_x']:6.2f} x {fit['spacing_y']:6.2f} um"
          f"   angle {fit['angle']:7.1f} deg   shear {fit['shear']:+5.1f} deg"
          f"   resid {fit['resid']:5.1f} um "
          f"({fit['resid']/max(fit['spacing'],1e-9):.2f} cell)"
          f"   {fit['n']} corner(s), {fit['mode']}, {fit['source']}"
          f"{'   REFLECTED' if fit['reflected'] else ''}")
    if fit['resid'] > 0.5 * fit['spacing']:
        print("           [WARNING] corners do not lie on one grid. Check the "
              "annotated bins and their indices.")
    if abs(fit['shear']) > 8.0:
        print("           [WARNING] grid is strongly sheared. Corners are "
              "probably mislabelled or misplaced.")
    if fit['spacing_x'] > 0 and fit['spacing_y'] > 0:
        ratio = max(fit['spacing_x'], fit['spacing_y']) / \
                min(fit['spacing_x'], fit['spacing_y'])
        if ratio > 1.25:
            print("           [WARNING] x and y cell sizes differ by more "
                  f"than 25% ({ratio:.2f}x).")



def assign_bins_to_grid_cells(bins_df, fit, n_x, n_y):
    ax, ay, origin = fit['ax'], fit['ay'], fit['origin']
    sx, sy = fit['spacing_x'], fit['spacing_y']

    x_unit = ax / max(np.linalg.norm(ax), 1e-12)
    y_unit = ay / max(np.linalg.norm(ay), 1e-12)

    ii, jj = np.meshgrid(np.arange(n_x), np.arange(n_y), indexing='ij')
    uu = ii.ravel() + 0.5
    vv = jj.ravel() + 0.5
    cen_xy = origin + uu[:, None] * ax + vv[:, None] * ay

    bin_xy = bins_df[['x', 'y']].values
    A = np.column_stack([ax, ay])
    if abs(np.linalg.det(A)) < 1e-9:
        print("  [ERROR] degenerate grid axes; skipping ROI.")
        empty = bins_df.copy()
        empty['assigned_x_idx'] = -1
        empty['assigned_y_idx'] = -1
        empty['dist_from_centroid_um'] = np.nan
        empty['within_footprint'] = False
        empty['low_confidence'] = False
        return empty, (ax, ay, origin), x_unit, y_unit

    uvb = np.linalg.solve(A, (bin_xy - origin).T).T
    xi = np.floor(uvb[:, 0]).astype(int)
    yi = np.floor(uvb[:, 1]).astype(int)

    inside = (xi >= 0) & (xi < n_x) & (yi >= 0) & (yi < n_y)
    xi_c, yi_c = xi.clip(0, n_x - 1), yi.clip(0, n_y - 1)
    dist = np.linalg.norm(bin_xy - cen_xy[xi_c * n_y + yi_c], axis=1)
    keep = inside & (dist <= MAX_CENTROID_DIST_FRAC * max(sx, sy))

    n_out = int((~keep).sum())
    if n_out:
        pct = 100.0 * n_out / len(bins_df)
        print(f"  Bins outside the fitted grid: {n_out:,} of "
              f"{len(bins_df):,} ({pct:.0f}%)")
        if pct > 50:
            print("    [WARNING] over half the ROI bins fall outside the grid. "
                  "The polygon is much larger than the indentation footprint, "
                  "or the corner annotations are wrong.")

    result = bins_df.copy()
    result['assigned_x_idx'] = np.where(keep, xi_c, -1)
    result['assigned_y_idx'] = np.where(keep, yi_c, -1)
    result['dist_from_centroid_um'] = np.where(keep, dist, np.nan)
    result['within_footprint'] = False
    result['low_confidence'] = False

    low_conf = empty = 0
    for cx in range(n_x):
        for cy in range(n_y):
            mask = keep & (xi_c == cx) & (yi_c == cy)
            idx = result.index[mask]
            if len(idx) == 0:
                empty += 1
                continue
            d = dist[mask]
            sel = idx[np.argsort(d)[:min(N_BINS_PER_BOX, len(idx))]]
            result.loc[sel, 'within_footprint'] = True
            if len(idx) < MIN_BINS_PER_BOX:
                result.loc[sel, 'low_confidence'] = True
                low_conf += 1

    if empty:
        print(f"  Cells with no bins: {empty} of {n_x * n_y}")
    if low_conf:
        print(f"  Low-confidence cells (< {MIN_BINS_PER_BOX} bins): {low_conf}")

    return result, (ax, ay, origin), x_unit, y_unit



def attach_mechanical_values(bins_assigned, nano_df):
    """
    Attach mechanical property values to each selected bin via
    (assigned_x_idx, assigned_y_idx) lookup in the nanoindentation CSV.
    Bins where within_footprint=False retain N/A for all mechanical cols.
    """
    available_src = [v for v in NANO_SCALAR_COLS.values()
                     if v in nano_df.columns]
    reverse_map   = {v: k for k, v in NANO_SCALAR_COLS.items()}
    result        = bins_assigned.copy()
 
    for src_col in available_src:
        result[reverse_map[src_col]] = np.nan
    result['valid_measurement'] = False
    result['nano_point_id']     = 'N/A'
    result['nano_x_orig'] = np.nan
    result['nano_y_orig'] = np.nan

    mask = result['within_footprint']
    if not mask.any():
        return result
 
    valid_keys = set(zip(
        nano_df.index.get_level_values('X_idx'),
        nano_df.index.get_level_values('Y_idx'),
    ))
 
    for i, (xi, yi) in zip(result.index[mask],
                            zip(result.loc[mask, 'assigned_x_idx'],
                                result.loc[mask, 'assigned_y_idx'])):
        key = (int(xi), int(yi))
        if key in valid_keys:
            row = nano_df.loc[key]
            for src_col in available_src:
                if src_col in row.index:
                    result.at[i, reverse_map[src_col]] = row[src_col]
            result.at[i, 'valid_measurement'] = True
            result.at[i, 'nano_x_orig'] = row['X_orig']
            result.at[i, 'nano_y_orig'] = row['Y_orig']
 
    result.loc[mask, 'nano_point_id'] = (
        'X' + result.loc[mask, 'assigned_x_idx'].astype(str) +
        'Y' + result.loc[mask, 'assigned_y_idx'].astype(str)
    )
    return result
 
 
def plot_verification(roi_bins, assigned_bins, x_unit, y_unit,
                      grid_extent, corner_positions, origin,
                      roi_label, n_x, n_y, out_path):
    """
    Verification plot:
      Light grey   = all ROI bins
      Steel blue   = selected bins (nearest N_BINS_PER_BOX per cell)
      Red outlines = grid cell boundaries (rotated rectangles)
      Red dot      = cell centroid (mean of all bins in cell)
      Gold star    = extrapolated X0Y0 (reference only)
      Magenta dot  = annotated corner bins
    """
    ax_v, ay_v, org_v = grid_extent
    ax_v = np.asarray(ax_v, float)
    ay_v = np.asarray(ay_v, float)
    org_v = np.asarray(org_v, float)
    x_min = y_min = 0.0
    cell_w = cell_h = 1.0
 
    def proj_to_visium(px, py):
        """Convert projection-space coords back to VisiumHD space."""
        return org_v + px * ax_v + py * ay_v
 
    fig, ax = plt.subplots(figsize=(14, 8))
 
    # Grey: all ROI bins
    ax.scatter(roi_bins['x'], roi_bins['y'],
               c='lightgrey', s=0.4, alpha=0.5,
               label='All ROI bins', rasterized=True)
 
    # Blue: selected bins
    within = assigned_bins[assigned_bins['within_footprint']]
    ax.scatter(within['x'], within['y'],
               c='steelblue', s=0.4, alpha=0.7,
               label=f'Selected bins ({len(within):,})', rasterized=True)
 
    # Red: grid cell outlines + centroids
    patches  = []
    cent_x, cent_y = [], []
    for xi in range(n_x):
        for yi in range(n_y):
            px0 = x_min + xi * cell_w
            px1 = x_min + (xi + 1) * cell_w
            py0 = y_min + yi * cell_h
            py1 = y_min + (yi + 1) * cell_h
            corners = np.array([
                proj_to_visium(px0, py0),
                proj_to_visium(px1, py0),
                proj_to_visium(px1, py1),
                proj_to_visium(px0, py1),
            ])
            patches.append(MplPolygon(corners, closed=True))
 
            # Centroid from selected bins
            sel_mask = ((assigned_bins['assigned_x_idx'] == xi) &
                        (assigned_bins['assigned_y_idx'] == yi) &
                        assigned_bins['within_footprint'])
            if sel_mask.any():
                cent_x.append(assigned_bins.loc[sel_mask, 'x'].mean())
                cent_y.append(assigned_bins.loc[sel_mask, 'y'].mean())
 
    pc = PatchCollection(patches, facecolor='none',
                         edgecolor='red', linewidth=0.5, alpha=0.55)
    ax.add_collection(pc)
    if cent_x:
        ax.scatter(cent_x, cent_y, c='red', s=8, zorder=5,
                   label=f'Cell centroids ({len(cent_x)})')
 
    # Gold star: X0Y0 reference
    ax.scatter(origin[0], origin[1], c='gold', s=80,
               marker='*', zorder=8, label='X0Y0 (ref)')
 
    # Magenta: annotated corners
    if corner_positions:
        cx_c = [p[0] for p in corner_positions.values()]
        cy_c = [p[1] for p in corner_positions.values()]
        ax.scatter(cx_c, cy_c, c='magenta', s=30,
                   marker='X', zorder=7, label='Annotated corners')
        for lbl, pos in corner_positions.items():
            ax.annotate(lbl.split('_')[-1], pos, fontsize=7,
                        color='magenta', xytext=(4, 4),
                        textcoords='offset points', fontweight='bold')
 
    # Label the 4 grid corners in projection space
    for (xi, yi), lbl in [((0,0),'X0Y0'), ((n_x-1,0),f'X{n_x-1}Y0'),
                           ((0,n_y-1),f'X0Y{n_y-1}'),
                           ((n_x-1,n_y-1),f'X{n_x-1}Y{n_y-1}')]:
        px = x_min + xi * cell_w
        py = y_min + yi * cell_h
        pos = proj_to_visium(px, py)
        ax.annotate(lbl, pos, fontsize=7, color='darkred',
                    xytext=(3, 3), textcoords='offset points')
 
    ax.set_aspect('equal')
    ax.set_xlabel('x (um, VisiumHD coordinates)')
    ax.set_ylabel('y (um, VisiumHD coordinates)')
    ax.set_title(
        f'ROI: {roi_label}  ({n_x} x {n_y} grid, '
        f'up to {N_BINS_PER_BOX} bins/cell)\n'
        f'Grey = all ROI bins  |  Blue = selected  |  '
        f'Red = grid cells + centroids'
    )
    ax.legend(markerscale=2, fontsize=8, frameon=True,
              loc='best', borderpad=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()
    print(f"  Saved: {out_path}")
 
# =============================================================================
# MAIN
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

def main():
    import os

    os.chdir(OUTPUT_DIR)  # redirect all relative file saves to the date folder

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_DIR / f"run_log_{timestamp}.txt"

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)

        try:
            print("=" * 60)
            print("SpatialX-NI ROI assignment run")
            print(f"Date/time: {datetime.now().isoformat(timespec='seconds')}")
            print(f"Metadata file: {METADATA_FILE}")
            print(f"Output table: {OUTPUT_CSV}")
            print(f"Output directory: {OUTPUT_DIR}")
            print(f"NI root: {NI_ROOT}")
            print(f"N_BINS_PER_BOX: {N_BINS_PER_BOX}")
            print(f"MIN_BINS_PER_BOX: {MIN_BINS_PER_BOX}")
            print(f"VISCOSITY_MAX: {VISCOSITY_MAX}")
            print(f"Run log: {log_path}")
            print("=" * 60)

            print(f"Loading: {METADATA_FILE}")
            meta = pd.read_csv(METADATA_FILE, sep=METADATA_SEP, low_memory=False)
            print(f" Shape: {meta.shape}")

            required = [COL_X, COL_Y, COL_ROI, COL_BOUNDARY]
            missing = [c for c in required if c not in meta.columns]
            if missing:
                raise ValueError(
                    f"Missing columns: {missing}\n"
                    f"Found: {meta.columns.tolist()}"
                )

            mech_out_cols = list(NANO_SCALAR_COLS.keys())
            new_cols = (
                mech_out_cols +
                [
                    'nano_point_id',
                    'nano_x_idx',
                    'nano_y_idx',
                    'nano_x_orig',
                    'nano_y_orig',
                    'dist_from_centroid_um',
                    'within_footprint',
                    'valid_measurement',
                    'low_confidence',
                ]
            )

            for col in new_cols:
                meta[col] = 'N/A'

            meta['within_footprint'] = 'False'
            meta['valid_measurement'] = 'False'
            meta['low_confidence'] = 'False'

            print("\n[1/3] Fitting grid geometry from corner annotations")
            GEOM = {}
            for _rk, _cfg in ROI_CONFIGS.items():
                _m = meta[COL_ROI] == _cfg['roi_label']
                if _m.sum() == 0:
                    GEOM[_rk] = dict(fit=None)
                    print(f"  {_rk:<8s} NO BINS for {COL_ROI} == "
                          f"'{_cfg['roi_label']}'")
                    continue
                _xy = meta.loc[_m, [COL_X, COL_Y]].values
                try:
                    _xu, _yu, _or, _cp, _pts, _bpts = recover_axes_from_corners(
                        meta,
                        to_internal_corners(_cfg['corner_annotations']),
                        _xy, _cfg['n_x'], _cfg['n_y'])
                except ValueError as _e:
                    print(f"  {_rk:<8s} ERROR: {_e}")
                    GEOM[_rk] = dict(fit=None)
                    continue
                GEOM[_rk] = dict(x_unit=_xu, y_unit=_yu, origin=_or,
                                 corner_positions=_cp, corner_pts=_pts,
                                 border_pts=_bpts,
                                 fit=solve_roi(_pts, _bpts,
                                               _cfg['n_x'], _cfg['n_y']))
                report_fit(_rk, GEOM[_rk]['fit'])

            print("\n[2/3] Resolving spacing per tissue")
            _sp = resolve_tissue_spacings(
                {k: v.get('fit') for k, v in GEOM.items()})
            for _rk, _g in GEOM.items():
                if _g.get('fit') is None and 'corner_pts' in _g:
                    _g['fit'] = fit_with_spacing(
                        _g['corner_pts'], _g['x_unit'], _g['y_unit'],
                        _sp.get(tissue_of(_rk)), _g.get('border_pts'),
                        ROI_CONFIGS[_rk]['n_x'], ROI_CONFIGS[_rk]['n_y'])
                    report_fit(_rk, _g['fit'])
                    if _g['fit'] is None:
                        print(f"  {_rk:<8s} [FATAL] no full corner and no "
                              f"tissue spacing available")

            print("\n[3/3] Assigning bins")
            for roi_key, cfg in ROI_CONFIGS.items():
                print(f"\n{'=' * 60}")
                print(f"ROI: {roi_key}")

                roi_mask = meta[COL_ROI] == cfg['roi_label']
                n_roi = roi_mask.sum()
                if n_roi == 0:
                    print(f" WARNING: No bins with {COL_ROI} == '{cfg['roi_label']}'.")
                    continue

                print(f" ROI bins: {n_roi:,}")

                roi_bins = meta.loc[roi_mask, [COL_X, COL_Y]].copy()
                roi_bins = roi_bins.rename(columns={COL_X: 'x', COL_Y: 'y'})
                roi_bins.index = meta.index[roi_mask]
                roi_bins_xy = roi_bins[['x', 'y']].values

                g = GEOM.get(roi_key)
                if g is None or g['fit'] is None:
                    print(f" ERROR: no grid solution for {roi_key}.")
                    continue
                x_unit, y_unit = g['x_unit'], g['y_unit']
                origin, corner_positions = g['origin'], g['corner_positions']

                print(
                    f" Assigning {n_roi:,} bins to "
                    f"{cfg['n_x']} x {cfg['n_y']} grid cells..."
                )

                assigned, grid_extent, x_unit, y_unit = assign_bins_to_grid_cells(
                    roi_bins,
                    g['fit'],
                    cfg['n_x'],
                    cfg['n_y'],
                )

                n_within = int(assigned['within_footprint'].sum())
                print(f" Bins selected (nearest {N_BINS_PER_BOX}/cell): {n_within:,}")
                print(f" Bins in ROI but not selected: {n_roi - n_within:,}")

                try:
                    nano_path = find_nano_csv(roi_key, cfg)
                except FileNotFoundError as e:
                    print(f" ERROR: {e}")
                    continue
                print(f" Loading: {nano_path.name}")
                nano_df = load_nano_csv(nano_path, sample_id=cfg.get('sample_id'))
                print(f" Valid indentation points: {len(nano_df)}")

                assigned = attach_mechanical_values(assigned, nano_df)
                n_valid = int(assigned['valid_measurement'].sum())
                print(f" Bins with valid mechanical values: {n_valid:,}")
                print(f" Bins selected but no measurement (N/A): {n_within - n_valid:,}")

                plot_verification(
                    roi_bins,
                    assigned,
                    x_unit,
                    y_unit,
                    grid_extent,
                    corner_positions,
                    origin,
                    roi_key,
                    cfg['n_x'],
                    cfg['n_y'],
                    f"verification_{roi_key}.png",
                )

                idx = assigned.index

                for col in mech_out_cols:
                    if col in assigned.columns:
                        meta.loc[idx, col] = (
                            assigned[col]
                            .where(assigned['within_footprint'], other='N/A')
                            .fillna('N/A')
                            .values
                        )

                meta.loc[idx, 'within_footprint'] = assigned['within_footprint'].astype(str).values
                meta.loc[idx, 'nano_x_idx'] = assigned['assigned_x_idx'].where(
                    assigned['within_footprint'], other='N/A').values
                meta.loc[idx, 'nano_y_idx'] = assigned['assigned_y_idx'].where(
                    assigned['within_footprint'], other='N/A').values
                meta.loc[idx, 'nano_x_orig'] = assigned['nano_x_orig'].where(
                    assigned['within_footprint'], other='N/A'
                ).values

                meta.loc[idx, 'nano_y_orig'] = assigned['nano_y_orig'].where(
                    assigned['within_footprint'], other='N/A'
                ).values
                meta.loc[idx, 'dist_from_centroid_um'] = (
                    assigned['dist_from_centroid_um'].where(
                        assigned['within_footprint'], other='N/A'
                    ).values
                )
                meta.loc[idx, 'valid_measurement'] = assigned['valid_measurement'].astype(str).values
                meta.loc[idx, 'low_confidence'] = assigned['low_confidence'].astype(str).values
                meta.loc[idx, 'nano_point_id'] = assigned['nano_point_id'].values

            meta = meta.fillna('N/A')
            meta.to_csv(OUTPUT_CSV, sep='\t', index=False)
            print(f"Saved: {OUTPUT_CSV} shape: {meta.shape}")

            # ------------------------------------------------------------------
            # EXTRA EXPORT: slim table for SpatialX re-import
            # Contains only unique barcodes + mechanical features
            # ------------------------------------------------------------------
            slim_cols = [UNIQUE_BARCODE_COL] + mech_out_cols
            slim = meta[slim_cols].copy()
            slim_out = OUTPUT_DIR / "metadata_nano_mech_for_spatialx.tsv"
            slim.to_csv(slim_out, sep="\t", index=False)
            print(f"Saved: {slim_out} shape: {slim.shape}")

            # ------------------------------------------------------------------
            # EXTRA EXPORT 1: nano_centroid.csv
            # One representative SpatialX 4um bin per matched nano grid point.
            # - keep only within_footprint == True
            # - keep only valid_measurement == True
            # - group by (ROIs, nano_point_id)
            # - within each group, keep the row with the smallest dist_from_centroid_um
            # ------------------------------------------------------------------
            cellcluster_col = "CellCluster"
            compartments_col = "Compartments"
            nano_x_orig_col = "nano_x_orig"
            nano_y_orig_col = "nano_y_orig"

            within_mask = meta["within_footprint"].astype(str).str.strip().str.lower() == "true"
            valid_mask = meta["valid_measurement"].astype(str).str.strip().str.lower() == "true"

            centroid_source = meta.loc[
                within_mask &
                valid_mask &
                meta["nano_point_id"].notna() &
                (meta["nano_point_id"].astype(str).str.strip() != "N/A")
            ].copy()

            centroid_source["dist_from_centroid_um_num"] = pd.to_numeric(
                centroid_source["dist_from_centroid_um"],
                errors="coerce"
            )

            centroid_idx = (
                centroid_source
                .groupby([COL_ROI, "nano_point_id"])["dist_from_centroid_um_num"]
                .idxmin()
            )

            nano_centroid = centroid_source.loc[centroid_idx].copy()

            nano_centroid_cols = [
                UNIQUE_BARCODE_COL,       # unique SpatialX barcode ID
                COL_X,                    # ST X coordinate
                COL_Y,                    # ST Y coordinate
                COL_ROI,                  # ROI ID
                nano_x_orig_col,          # NI X coordinate
                nano_y_orig_col,          # NI y coordinate
                "Elasticity_l_Pa",        # Stiffness values
                "dist_from_centroid_um",  # subset using lowest value indicate centroid position
            ]

            nano_centroid_cols_existing = [
                c for c in nano_centroid_cols if c in nano_centroid.columns
            ]
            nano_centroid = nano_centroid[nano_centroid_cols_existing].copy()

            nano_centroid_out = OUTPUT_DIR / "Nano_centroid.csv"
            nano_centroid.to_csv(nano_centroid_out, index=False)
            print(f"Saved: {nano_centroid_out} shape: {nano_centroid.shape}")

            # ------------------------------------------------------------------
            # EXTRA EXPORT 2: Celltype_coord.csv
            # Keep only the within-footprint subset.
            # ------------------------------------------------------------------

            celltype_source = meta.loc[within_mask].copy()

            celltype_coord_cols = [
                UNIQUE_BARCODE_COL,       # unique SpatialX barcode ID
                COL_X,                    # ST X coordinate
                COL_Y,                    # ST Y coordinate
                nano_x_orig_col,          # NI X coordinate
                nano_y_orig_col,          # NI y coordinate
                COL_ROI,                  # ROI ID
                cellcluster_col,          # "CellCluster"
                compartments_col,         # "Compartments"
                "dist_from_centroid_um",  # distance from centroid
            ]

            celltype_coord_cols_existing = [
                c for c in celltype_coord_cols if c in celltype_source.columns
            ]
            celltype_coord = celltype_source[celltype_coord_cols_existing].copy()

            celltype_coord_out = OUTPUT_DIR / "Celltype_coord.csv"
            celltype_coord.to_csv(celltype_coord_out, index=False)
            print(f"Saved: {celltype_coord_out} shape: {celltype_coord.shape}")

            print(f"\n{'=' * 60}")
            print(f"Saved: {OUTPUT_CSV} shape: {meta.shape}")
            for c in new_cols:
                print(f"  {c}: {(meta[c] != 'N/A').sum():,} non-N/A values")
            print("=" * 60)
            print(f"Run log saved to: {log_path}")

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
 
if __name__ == "__main__":
    main()