"""
12_Corr_stiffness_expression_correlation.py

Correlates gene expression with stiffness at the 50 um nanoindentation
measurement, across whole tissue and within each compartment, pooled and
stratified by age.

Parameters
    Unit of observation is the 50 um measurement, not the 4 um bin. Stiffness
    is constant within a measurement, so the ~92 bins sharing one are
    pseudoreplicates. Bins are summed into their measurement and normalised
    once at that level.

    AGE_AS_COVARIATE            True. Run once False and once True; the two
                                write to sibling folders (_noAgeAdj and
                                _ageAdj) so neither overwrites the other
    COMPARTMENT_MIN_FRAC        0.50 of a measurement's bins must be in
                                compartment C for it to enter that analysis.
                                A 50/50 measurement appears in both, carrying
                                the same stiffness into each, with expression
                                aggregated from that compartment's bins only
    MIN_POINTS_PER_COMPARTMENT  60
    MIN_POINTS_PER_STRATUM      30
    DETECT_FRAC                 0.10, gene must be detected in at least 10% of
                                measurements, and exceed GENE_MIN_TOTAL_COUNT
    FDR_THRESH, RHO_THRESH      0.05 and 0.1
    GSEA_MIN_SIZE, MAX_SIZE     10 and 500
    GSEA_PERM                   1000
    Only Elasticity_l_Pa is analysed.

    Sequencing depth is a covariate, not a selection criterion. Selecting on
    detection makes the log-normalised value a function of 1/nCount, which
    itself correlates with stiffness, so every gene inherits that correlation.
    Depth is partialled out by rank residualisation instead. Script 15
    demonstrates the failure mode.

    Five statistics per gene, each with its own BH-FDR:
        rho_global         depth held constant, plus age if AGE_AS_COVARIATE
        rho_within_animal  depth and animal identity held constant, strictest
        rho_young          young measurements only, depth and animal constant
        rho_aged           aged measurements only, same
        rho_delta          rho_aged minus rho_young, Fisher z test
    The stratified statistics exist because pooling can reverse a sign.
    Aldh1a1 is -0.130 pooled but +0.018 in young and -0.061 in aged.

Actions required
    Set AGE_AS_COVARIATE = False, run, then set it True and run again.
        py 12_Corr_stiffness_expression_correlation.py

Input
    expression_export/ from script 10, containing
        step9_expression_matrix.mtx        log-normalised, counts reconstructed
        step9_counts_matrix.mtx            used directly if present
        step9_expression_barcodes.txt
        step9_expression_features.txt
        step9_expression_metadata_matched.tsv
    metadata_with_nano.tsv from script 9
    mechanomics_gene_set.csv
Output
    pointlevel_expression_outputs_<ageAdj|noAgeAdj>/Elasticity_l_Pa/
        <group>_..._volcano_table_all_genes.csv       every gene, rho and FDR.
                                                      Read by script 13
        <group>_..._significant_only_r_values.csv     the subset passing both
                                                      thresholds
        <group>_..._pointlevel_correlations.csv       per-measurement values
        <group>_..._prerank_gsea_results.csv          curated by script 11
        <group>_..._matrisome_enrichment.csv          Fisher exact, background
                                                      is the genes tested in
                                                      that group
        volcano and scatter figures
    Groups are whole, CL, S, V and F.
    Produces Fig. 3 and Fig. 4e to k.

Note
    The Ovarian Mechanomics set holds 1,724 genes present in the matrix, which
    exceeds GSEA_MAX_SIZE, so it is not tested as a term. It is used to outline
    points on the volcano.

Dependencies
    pip install pandas numpy scipy statsmodels matplotlib seaborn gseapy
                adjustText
"""

import os
import textwrap

# BLAS threads must be pinned before numpy is imported, otherwise the rank
# residualisation loop oversubscribes every core.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# gseapy is only needed for the preranked GSEA. It is imported defensively so a
# machine without it can still run every correlation, volcano, heatmap and
# scatter panel; only the pathway section is skipped, and the run log says so.
# Install it with:  pip install gseapy
try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
    GSEAPY_IMPORT_ERROR = ""
except Exception as _gsea_err:          # ModuleNotFoundError, and any
    gp = None                           # half-installed dependency of it
    GSEAPY_AVAILABLE = False
    GSEAPY_IMPORT_ERROR = f"{type(_gsea_err).__name__}: {_gsea_err}"

from adjustText import adjust_text
from scipy import sparse, io as sio
from scipy.stats import rankdata, spearmanr, fisher_exact
from scipy.stats import t as student_t, norm as normal_dist
from statsmodels.stats.multitest import multipletests

import colorsys
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

warnings.filterwarnings("ignore")

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# THE ONE SWITCH TO FLIP BETWEEN RUNS
#
# False -> rho_global adjusts for sequencing depth only.
# True  -> rho_global adjusts for sequencing depth AND age group, so a gene
#          that merely differs between young and aged tissue no longer scores.
# The output folder name changes with it, so the first run is never clobbered.
# -----------------------------------------------------------------------------
AGE_AS_COVARIATE = True

# -----------------------------------------------------------------------------
# WHERE THE DATA LIVES
#
# One place to change when you move workstations. The candidates are tried in
# order and the FIRST ONE THAT EXISTS wins, so the same file runs on either
# machine without editing. Put the drive you use most often first.
#
# Set PROJECT_ROOT to a path directly to override the search entirely.
# -----------------------------------------------------------------------------
PROJECT_ROOT_CANDIDATES = [
    r"Y:\biejly\huanting\Visium_Data\001_SpatialX",
    r"Z:\Visium_Data\001_SpatialX",
    r"/Users/turleyjm/Mirror/MBI coding/Nano-omics",
]
PROJECT_ROOT = None          # None -> first candidate that exists

# The sub-path from the project root down to the metadata TSV. Change this only
# if the folder layout differs between the two drives, which is ASSUMED HERE
# NOT TO BE THE CASE. If Y: is laid out differently, edit this line.
METADATA_RELPATH = Path("20260803_FinalAnnotatedMetadata") / "metadata_with_nano.tsv"


def _resolve_project_root():
    """
    First candidate that exists, or a clear failure naming every path tried.

    Failing loudly here is deliberate. A missing drive would otherwise surface
    much later as a confusing 'file not found' on the metadata TSV, or worse,
    as a silently empty analysis.
    """
    if PROJECT_ROOT is not None:
        return Path(PROJECT_ROOT)
    for cand in PROJECT_ROOT_CANDIDATES:
        if Path(cand).exists():
            return Path(cand)
    raise FileNotFoundError(
        "None of the project roots exist. Tried:\n  "
        + "\n  ".join(str(c) for c in PROJECT_ROOT_CANDIDATES)
        + "\n\nMap the drive, or set PROJECT_ROOT at the top of this script "
          "to the folder that holds 20260803_FinalAnnotatedMetadata."
    )


PROJECT_ROOT_RESOLVED = _resolve_project_root()
print(f"Project root: {PROJECT_ROOT_RESOLVED}")

# Script 6/7 output metadata TSV
METADATA_TSV = PROJECT_ROOT_RESOLVED / METADATA_RELPATH

# Script 9 output folder
EXPR_DIR = METADATA_TSV.parent / "expression_export"

# Script 10 output folder (suffix set by AGE_AS_COVARIATE)
OUTPUT_DIR = METADATA_TSV.parent / (
    "pointlevel_expression_outputs" + ("_ageAdj" if AGE_AS_COVARIATE else "_noAgeAdj")
)

# Primary mechanical target. Elasticity_l_Pa only, by design.
PRIMARY_TARGETS = ["Elasticity_l_Pa", "Elasticity_eff_Pa", "Loss", "Viscosity_Ns_m"]
LOG_TARGETS = {"Elasticity_l_Pa", "Elasticity_eff_Pa"}
LOG_TARGET = True

# -----------------------------------------------------------------------------
# POINT CONSTRUCTION
#
# nano_point_id is a grid coordinate such as X3_Y5 and repeats in every ROI and
# every slice. Grouping on it alone merged 2,927 real points into 521 groups,
# 82% of which then held more than one stiffness value. Always use the
# composite key.
# -----------------------------------------------------------------------------
POINT_KEY_COLS = ["Tissue.slices", "ROIs", "nano_point_id"]
MIN_BINS_PER_POINT = 5
SCALE_FACTOR = 1e4

# Compartment membership: fraction of the point's 4 um bins in that compartment.
COMPARTMENT_MIN_FRAC = 0.50
COMPARTMENTS_TO_RUN = ["S", "F", "CL", "V"]
MIN_POINTS_PER_COMPARTMENT = 60
MIN_POINTS_PER_STRATUM = 30

# -----------------------------------------------------------------------------
# GENE FILTERING
#
# "compartment" recomputes detection inside each compartment, which keeps
# vasculature-restricted genes testable and keeps the BH correction honest for
# that compartment. "global" uses one gene set defined on all points, which
# makes the compartments directly comparable but dilutes power in the small
# ones. Change this if you want the comparable version.
# -----------------------------------------------------------------------------
GENE_FILTER_SCOPE = "compartment"      # "compartment" or "global"
GENE_MIN_DETECT_FRAC = 0.10            # detected in >= 10% of points
GENE_MIN_TOTAL_COUNT = 20              # >= 20 counts summed over points

# -----------------------------------------------------------------------------
# PER-GENE POINT SUBSETTING FOR CORRELATIONS
#
# Genes are still screened on ALL retained points using the filter above.
# Correlations are then computed only on the subset of points where the gene is
# expressed strongly enough to count as detected at point level.
# -----------------------------------------------------------------------------
EXPRESSION_SUBSET_MODE = "raw_count"   # currently only "raw_count"
EXPRESSION_SUBSET_MIN_COUNT = 1         # point-level summed counts for a gene
MIN_POINTS_PER_GENE_CORR = 30           # minimum points for global / within

# -----------------------------------------------------------------------------
# SIGNIFICANCE
#
# RHO_THRESH is 0.05, not 0.3. 10d measured the composition ceiling at
# rho = 0.075 (stroma fraction) and the largest observed per-gene rho was
# 0.126, so 0.3 would return nothing by construction. Report rho, not just FDR.
# -----------------------------------------------------------------------------
FDR_THRESH = 0.05
RHO_THRESH = 0.1
RHO_DIFF_THRESH = 0.1
TOP_N_LABEL = 20
# 10 labels per direction need ~80 mm of horizontal room at 5 pt. In a 35 mm
# panel they collide however hard adjust_text works. 4 is what fits.
MECHANOMICS_LABEL_N = 4                # top N each direction, Mechanomics only

# Which statistic feeds GSEA and the matrisome test.
RANK_COL_FOR_PATHWAYS = "rho_global"

# ---- Which statistics get scatter panels, GSEA and the enrichment tests ----
# per_stat already holds the volcano output for EVERY entry of VOLCANO_STATS,
# for every group including each compartment, so adding statistics here costs
# no recomputation at all: only extra figures and tables.
#
# The point of listing "young" and "aged" is that CL-within-young and
# CL-within-aged are different tests from CL-overall. Pooling them and then
# labelling a panel with two stratum correlations invites the reader to compare
# numbers that came from different models.
#
# None = fall back to RANK_COL_FOR_PATHWAYS alone, the previous behaviour.
DOWNSTREAM_STATS = ["global", "young", "aged"]

# Outputs for "global" keep their existing filenames so nothing you have
# already curated is renamed or overwritten; the other statistics get the stat
# appended. Set to False to put the stat in every name, which is tidier but
# renames your existing global figures.
#
# Now True. With the display names below, leaving it False would produce
# "whole_correlation" for the pooled statistic and "whole_youngcorr_
# correlation" for the stratum one, so the pooled files would be the only ones
# without a statistic in the name, which is the ambiguity you are trying to
# remove.
STAT_SUFFIX_ON_GLOBAL = True

# -----------------------------------------------------------------------------
# DISPLAY NAMES FOR FOLDERS, FILENAMES AND TITLES
# -----------------------------------------------------------------------------
# "global" was doing two unrelated jobs: the group label for whole tissue, and
# the key for the pooled statistic. A file called
# 3_global_correlation_..._global was unreadable for that reason.
#
# These maps rename what is WRITTEN. They do NOT rename anything the code
# looks up. models["global"], res["rho_global"], p_global, fdr_global,
# RANK_COL_FOR_PATHWAYS and every column in every CSV keep their existing
# names, so script 12, script 13 and script 14 keep working untouched.
#
# What DOES change on disk: the whole-tissue output folder becomes
# whole_<target> instead of global_<target>, and every file inside it is
# renamed. Any hard-coded path to global_Elasticity_l_Pa in another script
# needs updating.
GROUP_LABEL_DISPLAY = {"global": "whole"}

STAT_LABEL_DISPLAY = {
    "global": "globalcorr",        # pooled, regardless of age
    "young": "youngcorr",          # young points only
    "aged": "agedcorr",            # aged points only
    "within_animal": "withincorr",
    "delta": "deltacorr",
}


def group_disp(group_label):
    """Folder- and title-facing name for a group. Internals are untouched."""
    return GROUP_LABEL_DISPLAY.get(str(group_label), str(group_label))


def stat_disp(stat):
    """Folder- and title-facing name for a statistic."""
    return STAT_LABEL_DISPLAY.get(str(stat), str(stat))


# The whole-tissue group is labelled once, here, and every folder name, CSV
# name, figure name and title downstream reads it from this one place. Anything
# that needs to ask "is this the whole-tissue run?" compares against this
# constant rather than against the literal "global", which is why renaming the
# group cannot silently break the comparison.
WHOLE_GROUP_LABEL = GROUP_LABEL_DISPLAY.get("global", "global")

# On a global panel, also draw dashed rho_young and rho_aged reference lines
# alongside the solid black rho_global one, so a gene selected globally shows
# at a glance whether the two age groups actually agree.
RHO_LINES_PER_STRATUM = True
RHO_LINE_LW = 1.6          # black selected-stat line
RHO_LINE_LW_STRATUM = 1.3  # dashed young / aged lines

# Which statistics get a volcano.
VOLCANO_STATS = ["global", "within_animal", "young", "aged", "delta"]

# -----------------------------------------------------------------------------
# PATHWAY ANALYSIS (GSEA only; ORA removed)
# -----------------------------------------------------------------------------
RUN_GSEA = True
GSEA_GENESETS = [
    "GO_Biological_Process_2025",
    "Reactome_2022",
]
GSEA_ORGANISM = "mouse"
GSEA_PERM = 1000
GSEA_MIN_SIZE = 10
GSEA_MAX_SIZE = 200

# -----------------------------------------------------------------------------
# WHICH COMPARISONS GET GSEA
#
# Seven runs in total, matching the seven comparisons you want:
#   whole        rho_global, rho_young, rho_aged        3
#   S, F, CL, V  rho_global each                        4
# Before this the compartments also ran young and aged, which was 15 runs, and
# the age-stratified compartment models rest on far fewer points.
# -----------------------------------------------------------------------------
GSEA_STATS_WHOLE = ["global", "young", "aged"]
GSEA_STATS_COMPARTMENT = ["global"]

# -----------------------------------------------------------------------------
# OVER-REPRESENTATION ANALYSIS (ORA)
#
# ORA asks a different question from GSEA. GSEA scores the WHOLE ranked list
# and asks whether a pathway's genes sit preferentially at one end. ORA takes
# the genes that already passed the significance gate and asks whether a
# pathway is over-represented among them, by a hypergeometric test. So ORA can
# find a pathway GSEA misses when a handful of that pathway's genes are strong
# hits but the rest of its genes sit randomly in the ranking, and GSEA can find
# a pathway ORA misses when many of its genes lean the same way without any of
# them individually clearing the gate.
#
# THE BACKGROUND IS THE WHOLE ANALYSIS. ORA compares "how many of my hits are
# in this pathway" against "how many of the UNIVERSE are in this pathway", so
# the universe decides the answer. Measured on a synthetic example with the
# same 32/104 overlap:
#     background  3,000 genes -> p = 1.3e-14, odds ratio  7.4
#     background 20,000 genes -> p = 1.6e-39, odds ratio 53.5
# Same data, twenty-five orders of magnitude apart. The correct universe here
# is the genes that ACTUALLY ENTERED the correlation test for this group and
# statistic, not the mouse genome, so that is what is passed as background.
# This is also why gp.enrich (local hypergeometric) is used and not
# gp.enrichr (the web service), which substitutes its own background.
# -----------------------------------------------------------------------------
RUN_ORA = True

# Which comparisons get ORA. Same seven-way split as GSEA by default, except
# that F is left out of ORA_COMPARTMENTS: add "F" if you want it.
ORA_STATS_WHOLE = ["global", "young", "aged"]
ORA_STATS_COMPARTMENT = ["global"]
ORA_COMPARTMENTS = ("CL", "S", "V")

# Positive and negative hits are tested as SEPARATE lists, never pooled. A
# pathway enriched among positively correlated genes and one enriched among
# negatively correlated genes are opposite findings; pooling them would let
# the two cancel or, worse, look like one result.
ORA_DIRECTIONS = ("positive", "negative")

# Gene sets are intersected with the background first, then filtered by size,
# so the sizes quoted are the sizes actually tested. Matches the GSEA bounds.
ORA_MIN_SET_SIZE = GSEA_MIN_SIZE
ORA_MAX_SET_SIZE = GSEA_MAX_SIZE

# Size bands reported on every ORA run, so a decision about widening the upper
# bound can be made from this experiment's own counts. Purely diagnostic: these
# numbers change nothing that is tested.
ORA_SIZE_REPORT_BANDS = (300, 500)

ORA_FDR_MAX = 0.05      # BH adjusted p value for calling a term
# Below this many genes in a direction, ORA has no power and is skipped with a
# note rather than run and reported. A 4-gene list can only ever produce
# terms driven by a single gene.
ORA_MIN_GENES = 10

# One combined dot plot per comparison. Positive-direction terms are drawn at
# +log2(odds ratio) and negative-direction terms at -log2(odds ratio), so the
# panel reads like the GSEA one. NOTE: this is a SIGNED ODDS RATIO, not an NES.
# The sign encodes which gene list the term came from; it is not a directional
# statistic computed within a single test.
ORA_COMBINED_DOTPLOT = True
# Terms with an odds ratio below 1 are UNDER-represented, not over-represented.
# On the combined panel the x sign is supposed to mean "which gene list", so a
# depleted term would be drawn on the wrong side and read as the opposite
# result. They are excluded from the figure and kept in the CSV, with a count
# printed so nothing disappears silently.
ORA_PLOT_ENRICHED_ONLY = True
ORA_OR_CLIP = 64.0      # odds ratio can be infinite when the non-hit count is
                        # zero; clipped so the axis stays readable

# -----------------------------------------------------------------------------
# THE RANKING METRIC
#
# This is the DEG analogy you asked for. In a DESeq2 -> fgsea workflow the
# ranking vector is the effect size per gene (log2FoldChange) and the adjusted
# p value is used to interpret the result, not to build the rank. Ranking by
# rho is the exact counterpart: rho IS the effect size here.
#
#   "rho"                 rank = rho.                 <- the log2FC analogue
#   "signed_neglog10_fdr" rank = sign(rho) * -log10(FDR).
#                         The direction comes from rho and the magnitude from
#                         significance. This is the analogue of ranking by
#                         sign(log2FC) * -log10(padj), which some fgsea
#                         vignettes use. It compresses effect size away: two
#                         genes at rho +0.30 and +0.05 with the same FDR get
#                         the same rank.
#   "rho_x_neglog10_fdr"  rank = rho * -log10(FDR). Uses both, but the product
#                         has no standard interpretation and is not what a
#                         DESeq2 workflow does.
#
# Left at "rho". If your R GSEA ranks by something else, set this to match
# before comparing the two, because the enrichment scores are not comparable
# across different ranking metrics.
GSEA_RANK_METRIC = "rho"
GSEA_RANK_FDR_FLOOR = 1e-300   # guards -log10(0) in the two FDR-based options

# Pathways drawn on the dot plot: this many in EACH direction, so 15 gives up
# to 15 positive NES + 15 negative NES on one panel. Applies to every group,
# whole and compartment alike.
PATHWAY_DOTPLOT_TOP_N_PER_DIRECTION = 15
# Kept only so old calls that pass top_n still work; top_n is now interpreted
# as a TOTAL and halved, exactly as before.
PATHWAY_DOTPLOT_TOP_N_GLOBAL = 2 * PATHWAY_DOTPLOT_TOP_N_PER_DIRECTION
PATHWAY_DOTPLOT_TOP_N_COMPARTMENT = 2 * PATHWAY_DOTPLOT_TOP_N_PER_DIRECTION
PATHWAY_DOTPLOT_TOP_N = PATHWAY_DOTPLOT_TOP_N_COMPARTMENT   # fallback default

# Width is FIXED for every GSEA bubble plot so the panels stack. Height is free
# and grows with the rows actually drawn, so a panel with 30 terms is not
# crushed into the same box as one with 6.
PATHWAY_DOTPLOT_W_MM   = 120     # same for every GSEA dot plot
PATHWAY_DOTPLOT_ROW_MM = 5.0     # was 4.0; 30 rows at 4 mm left no room
PATHWAY_DOTPLOT_PAD_MM = 20.0    # was 12.0; title, x label and legend
PATHWAY_DOTPLOT_H_MIN_MM = 45.0  # floor, so a 4-term panel is not a sliver
PATHWAY_DOTPLOT_H_MAX_MM = 260.0 # ceiling, so nothing runs off a page
PATHWAY_DOTPLOT_TERM_CHARS = 42  # term labels are truncated to this
PATHWAY_DOTPLOT_FDR_MAX = 0.05

# ---- explicit layout, in mm ------------------------------------------------
# The figure is laid out by hand rather than by constrained layout, because the
# colour bar and the bubble key have to occupy a fixed column on the right and
# split the figure height between them. Constrained layout sizes those around
# whatever the labels need, so no two panels came out the same.
#   LEFT    term labels
#   RIGHT   the colour bar + bubble key column, including their own labels
#   BOTTOM  x tick labels and the x axis label
#   TOP     the title
PATHWAY_DOTPLOT_LEFT_MM   = 50.0
PATHWAY_DOTPLOT_RIGHT_MM  = 26.0
PATHWAY_DOTPLOT_BOTTOM_MM = 12.0
PATHWAY_DOTPLOT_TOP_MM    = 8.0
PATHWAY_DOTPLOT_GAP_MM    = 3.0   # axes to colour bar
PATHWAY_DOTPLOT_CBAR_W_MM = 3.5   # colour bar width

# The right column is split in two: colour bar in the top half, bubble key in
# the bottom half, each taking half the FIGURE height.
PATHWAY_DOTPLOT_SPLIT_FRAC = 0.50
PATHWAY_DOTPLOT_COL_PAD_FRAC = 0.05   # breathing room at the top and bottom

# Zero stays in the middle of the x axis, so a panel with only negative NES
# still reads as "everything is on the down side" rather than filling the
# frame. Set False to let matplotlib scale to the data.
PATHWAY_DOTPLOT_SYMMETRIC_X = True

# ---- bubble size: three FIXED bins, identical on every panel ----------------
# The old code scaled bubble area between the smallest and largest set on that
# panel, so the same 40-gene pathway drew at a different size on every figure
# and the sizes could not be compared across panels. Now the bins are absolute.
#   small   < 15 genes
#   medium  15 to 49 genes
#   large   >= 50 genes
PATHWAY_DOTPLOT_SIZE_BREAKS = (15, 50)          # lower edges of medium, large
PATHWAY_DOTPLOT_SIZE_AREAS  = (25.0, 80.0, 175.0)   # matplotlib s, points^2
PATHWAY_DOTPLOT_SIZE_LABELS = ("< 15", "15 to 49", ">= 50")

# ---- colour scale ----------------------------------------------------------
# -log10(FDR). The floor is 1 rather than -log10(0.05) = 1.301 so the axis
# starts on a round number; nothing is plotted below 1.301 anyway, because the
# FDR filter is applied first. Anything at or above the ceiling is drawn in the
# darkest red and the top tick says so.
PATHWAY_DOTPLOT_CMIN = 1.0
PATHWAY_DOTPLOT_CMAX = 5.0
# The "*" that used to be appended to a capped term label. Redundant now that
# the colour bar itself reads ">= 5", and at a ceiling of 5 most terms are
# capped, so it would mark almost every row.
PATHWAY_DOTPLOT_MARK_CAPPED = False

# -----------------------------------------------------------------------------
# REFERENCE GENE LISTS
# -----------------------------------------------------------------------------
REFERENCE_DIR = PROJECT_ROOT_RESOLVED   # follows the drive resolved above,
                                       # so the matrisome, connectomeDB and
                                       # GOI files move with it

# Naba lab mouse matrisome masterlist. Its column headers differ between
# releases ("Division"/"Category" in the 2012 masterlist,
# "Matrisome Division"/"Matrisome Category" in some redistributions), so the
# loader auto-detects and prints what it found. Set these explicitly to
# override.
MATRISOME_CSV = REFERENCE_DIR / "Mm_Matrisome_Masterlist_Naba et al_2012.csv"
RUN_MATRISOME = True
MATRISOME_GENE_COL = None          # None -> auto-detect
MATRISOME_CATEGORY_COL = None      # None -> auto-detect ("Category")
MATRISOME_SUBCATEGORY_COL = None   # None -> auto-detect ("Division")

# Ligand-receptor pairs. connectomeDB stores these as "Ligand gene symbol" /
# "Receptor gene symbol"; both are recognised by the auto-detector.
LR_PAIRS_CSV = REFERENCE_DIR / "ConnectomeDB2025_mouse.csv"
LR_LIGAND_COL = None      # None -> auto-detect from common column names
LR_RECEPTOR_COL = None

# Custom genes-of-interest list. A one-column CSV of symbols, or a table with a
# gene / symbol column; both work. If the file is missing, the hardcoded
# GENES_OF_INTEREST_FALLBACK below is used instead.
GOI_CSV = REFERENCE_DIR / "ListOfGOI-20240923.csv"
GOI_GENE_COL = None       # None -> auto-detect

# Mechanomics = matrisome UNION genes-of-interest. connectomeDB contributes
# ligand-receptor ANNOTATION only, so a gene found nowhere but connectomeDB is
# not added (that would import ~873 genes you never chose). Set this True to
# add those partners back; they are then coded 'C' in the source column.
MECHANOMICS_ADD_LR_PARTNERS = False

# -----------------------------------------------------------------------------
# GENES OF INTEREST
#
# Two separate lists, deliberately:
#   GENES_OF_INTEREST      loaded from GOI_CSV. Feeds the Mechanomics set. Can
#                          be as long as you like.
#   MANUAL_GENE_PLOT_LIST  the curated panel list below. One scatter figure is
#                          drawn per gene, so this one stays short.
#
# Ctgf and Cyr61 were renamed years ago and are NOT in current references.
# The current symbols are Ccn2 and Ccn1.
# -----------------------------------------------------------------------------
GENES_OF_INTEREST_FALLBACK = [
    "Col1a1", "Col1a2", "Col3a1", "Col13a1", "Col14a1",
    "Col16a1", "Col18a1", "Col4a1", "Col4a2", "Col4a3", "Col4a4",
    "Emilin2", "Fn1", "Habp2", "Hspg2",
    "Lama1", "Lamb1", "Lamc1", "Cd44",
    "Eln", "Fbn1", "Fbn2", "Postn",
    "Loxl1", "Tgfbi", "Vcan", "Mmp13", "Mmp9",
    "Sparc", "Tnc", "Thbs1", "Thbs2", "Tagln",
    "Yap1", "Wwtr1", "Tead1", "Piezo1", "Piezo2",
    "Cdkn1a", "Trp53", "Ptger4", "Ptn", "Egfr",
    "Ednra", "Htr2a", "Icam1", "Pdgfb",
    "Cdh1", "Cdh2", "Cnn2",
    "Acta2", "Vcl", "Myh2",
    "Itga1", "Itga2", "Itga2b", "Itga4", "Itga5", "Itga6",
    "Itgav", "Itgb1", "Itgb2", "Itgb3", "Itgb5",
    "Ccn2", "Ccn1", "Axl", "Birc5", "Ccna2", "Ctcf", "Ets1",
    "Fgfr2", "Fgfr4", "Fosl1", "Jun", "Kif14", "Plk1", "Pola1", "Slc2a3",
]

# Filled at run time from GOI_CSV, falling back to the list above.
GENES_OF_INTEREST = list(GENES_OF_INTEREST_FALLBACK)

# Scatter panels are drawn ONLY for significant Mechanomics hits: the top N by
# rho in each direction. MANUAL_GENE_PLOT_LIST is no longer used to choose
# panels. It used to be the starting point, which meant the 70 curated genes
# filled all 40 slots and not one actual hit was ever drawn.
PANEL_TOP_N_PER_DIRECTION = 50   # per direction, per statistic; ignored
                                 # when PANEL_ALL_SIGNIFICANT is True

# Draw a panel for EVERY gene that passes FDR_THRESH and RHO_THRESH, so the
# figures match the significant_only_r_values.csv row for row.
#   - drops the Mechanomics membership filter
#   - drops the top-N-per-direction cap
# This produces THOUSANDS of files. The count is printed before anything is
# drawn, and PANEL_CONFIRM_ABOVE stops the run so you can check rather than
# discovering it an hour later.
# True  = every gene past the gate, matching significant_only_r_values.csv
#         row for row. Thousands of files; see PANEL_CONFIRM_ABOVE.
# False = top PANEL_TOP_N_PER_DIRECTION per direction, per statistic.
PANEL_ALL_SIGNIFICANT = False

# Independent of the above: restrict panels to the Mechanomics set.
PANEL_REQUIRE_MECHANOMICS = False

# Plot the UNION of genes selected by any statistic for EVERY statistic, so a
# gene picked up in agedcorr also gets its globalcorr panel. Without this the
# three lists are built independently and are largely disjoint, which is why
# Lhcgr had an agedcorr panel and no globalcorr one.
PANEL_GENES_UNION_ACROSS_STATS = True

# The union above has one unwanted consequence: a statistic with ZERO hits of
# its own still got a full set of panels, carried in from the other two. So
# CL agedcorr, which passes nothing, was drawing hundreds of figures for genes
# that are not agedcorr hits at all. With this True a statistic that selected
# no genes of its own draws nothing, and a statistic that selected at least
# one draws the union as before.
PANEL_SKIP_STAT_WITH_NO_HITS = True

MANUAL_GENE_PLOT_LIST = list(GENES_OF_INTEREST_FALLBACK)   # reference only
MAX_GENE_PANELS = 100000      # ceiling on files per group; was 40
PANEL_CONFIRM_ABOVE = 2000    # abort with a message above this many panels
                              # per group. Raise it once you have seen the
                              # count and are happy to wait.
REFERENCE_TABLE_MAX_GENES = 100
EXPORT_LONG_TABLE = False

# Metadata columns
CLUSTER_COL = "CellCluster"
AGE_COL = "AgeGroup"
TISSUE_COL = "Tissue.slices"
NANO_ID_COL = "nano_point_id"
COMPARTMENT_COL = "Compartments"
ROI_COL = "ROIs"
GRID_X_COL = "nano_x_idx"
GRID_Y_COL = "nano_y_idx"
COL_X = "x"
COL_Y = "y"
DEPTH_COL = "nCount_RNA"

RANDOM_STATE = 42

# Script 9 export filenames
EXPR_MATRIX_FILE = "step9_expression_matrix.mtx"
EXPR_COUNTS_FILE = "step9_counts_matrix.mtx"
EXPR_BARCODES_FILE = "step9_expression_barcodes.txt"
EXPR_FEATURES_FILE = "step9_expression_features.txt"
EXPR_MATCHED_METADATA_FILE = "step9_expression_metadata_matched.tsv"

# Plot colours
YOUNG_COL = "#007A7A"
AGED_COL = "#B57BEE"
SOFT_COL = "#56B4E9"
STIFF_COL = "#E69F00"
CMAP_STIFF = "YlOrRd"

# -----------------------------------------------------------------------------
# DOT SIZES
#
# matplotlib's `s` is AREA in points squared, NOT diameter. Halving the
# diameter therefore means QUARTERING `s`, not halving it. Every marker size in
# this script now derives from POINT_SIZE so they cannot drift apart again.
#
#   POINT_SIZE      4.0   raw correlation panel (only drawn when
#                         PLOT_RAW_PANEL = True)
#   POINT_SIZE / 2  2.0   the corrected correlation panel, which is the one you
#                         actually get, and the size you said looks right.
#                         Volcanoes and the aged-vs-young scatter now match it.
#
# For dots half as WIDE as the current volcano dots rather than matching the
# correlation panel, use POINT_SIZE / 4 instead. That is smaller than the
# correlation dots, so the two figures would no longer agree.
# -----------------------------------------------------------------------------
POINT_SIZE = 4
POINT_ALPHA_FILL = 0.35
POINT_ALPHA_EDGE = 0.85
POINT_LINEWIDTH = 0.2

MARKER_SIZE_SMALL = POINT_SIZE / 2.0        # the shared "small dot" size


def ring_size_for(dot_size, gap_pt, lw_pt):
    """
    Scatter `s` for a hollow ring that sits just OUTSIDE a filled dot of area
    `dot_size`, with `gap_pt` points of clear space between the dot edge and
    the ring's inner edge.

    Written because every ring in this script used to be a hand-picked
    multiple of the dot area, and matplotlib's `s` is AREA in points squared,
    not diameter. So when the dots were halved in diameter (a QUARTER of the
    area) the fixed multiples did not follow, and the rings ended up more than
    twice the width of the dot they were meant to mark.

    Areas do not add the way widths do, so the arithmetic is done in diameter
    and converted back once:
        d_dot         = 2 * sqrt(dot_size / pi)
        d_centreline  = d_dot + 2 * gap_pt + lw_pt   (lw straddles the path)
        s             = pi * (d_centreline / 2) ** 2
    """
    d_dot = 2.0 * np.sqrt(max(float(dot_size), 0.0) / np.pi)
    d_centre = d_dot + 2.0 * float(gap_pt) + float(lw_pt)
    return float(np.pi * (d_centre / 2.0) ** 2)


# Clear space between a dot's edge and the ring drawn around it, in points.
# 0.30 pt keeps the ring hugging the dot at the sizes used here; raise it if
# you ever enlarge POINT_SIZE and want the rings to stay obvious.
RING_GAP_PT = 0.30

# Volcano
VOLCANO_POINT_SIZE_NS = MARKER_SIZE_SMALL
VOLCANO_POINT_SIZE_SIG = MARKER_SIZE_SMALL
# Left deliberately EQUAL to the dot area. The 0.2 pt stroke then straddles
# the dot's own edge, so a Mechanomics gene reads as a dark-rimmed dot rather
# than a dot inside a separate circle. To make it a true outer ring instead:
#   MECHANOMICS_OUTLINE_SIZE = ring_size_for(VOLCANO_POINT_SIZE_SIG,
#                                            RING_GAP_PT, MECHANOMICS_OUTLINE_LW)
MECHANOMICS_OUTLINE_LW = 0.2
MECHANOMICS_OUTLINE_SIZE = MARKER_SIZE_SMALL
VOLCANO_LEGEND_SIZE_MULT = 4.0   # legend swatches only; the plotted dots are
                                 # too small to read in a key

# Aged-vs-young rho scatter (compare_aged_young). These were hardcoded at
# 10 / 14 / 36, so they stayed large when everything else shrank.
# Both now sit at MARKER_SIZE_SMALL, exactly the volcano dot size. The
# significant dots used to be 1.4x larger, which made the two figures
# disagree about what a "dot" is.
DELTA_POINT_SIZE_NS = VOLCANO_POINT_SIZE_NS
DELTA_POINT_SIZE_SIG = VOLCANO_POINT_SIZE_SIG
# The ring around a labelled gene. Was MARKER_SIZE_SMALL * 3.6 with a 0.4 pt
# stroke: outer diameter 3.43 pt against a 1.60 pt dot, so the ring was 2.15x
# the width of the thing it was pointing at. Derived now, so it tracks the dot
# automatically: 2.80 pt outer, 1.75x the dot.
DELTA_LABEL_RING_LW = 0.3
DELTA_LABEL_RING_SIZE = ring_size_for(DELTA_POINT_SIZE_SIG, RING_GAP_PT,
                                      DELTA_LABEL_RING_LW)

# -----------------------------------------------------------------------------
# AGED VS YOUNG PANEL (step 7) FORMATTING
#
# The panel is 1.5 x 1.5 in (38 mm) and is saved with bbox_inches=None
# (EXACT_FIGURE_SIZE), so anything that does not fit is CLIPPED, not resized.
# The old version overflowed for one reason: ax.legend() with the labels
# "Not differentially correlated" and "Differentially correlated (n=130)".
# At 6 pt that legend measures about 2.5 in wide, and constrained layout
# treats an in-axes legend as furniture it must make room for, so it shrank
# the axes to 39% of the width and pushed it to x0 = 0.58 of the canvas.
# That is why the y label and the tick labels ran off the left edge.
# It is replaced below by an in-axes TEXT key, which layout ignores.
# -----------------------------------------------------------------------------
DELTA_FIG_SIZE = (1.5, 1.5)   # unchanged; do not grow the panel
DELTA_TITLE_FONT = 5.0        # was BASE_FONT (6)
DELTA_AXIS_FONT = 5.0         # was AXIS_FONT (6)
DELTA_TICK_FONT = 5.0         # was TICK_FONT (6)
DELTA_KEY_FONT = 4.5          # in-axes text key, replaces ax.legend()
DELTA_GENE_LABEL_FONT = 4.0   # was VOLCANO_LABEL_FONTSIZE (5)
DELTA_MAX_TICKS = 3           # 4 tick labels do not fit on a 25 mm axis
DELTA_TITLE_WRAP = 30         # characters per title line at DELTA_TITLE_FONT
DELTA_KEY_WRAP = 24           # characters per key line at DELTA_KEY_FONT

# Axis titles. The target name is in the plot title, so the axes only have to
# say which stratum the rho came from.
DELTA_XLABEL = "r_Young"
DELTA_YLABEL = "r_Aged"

# The key sits inside the axes, so blank space is reserved for it by raising
# the top of the y range by this fraction of the data range BEFORE anything is
# drawn. Without it the key text lands on the dot cloud.
DELTA_KEY_HEADROOM = 0.22
DELTA_KEY_SHOW_OTHER = False   # add a grey "other n=" line to the key

# The n= key and 20 gene labels cannot share a 25 mm axis. On the labelled
# panel the key is dropped; the counts are on the sibling panel.
DELTA_KEY_ON_LABELLED = False

# Head-room added to the data limits before adjust_text runs, as a fraction of
# the data range. Without it the outermost labels are pushed past the axes and
# clipped at the canvas edge.
DELTA_LABEL_MARGIN = 0.28

# Number of genes labelled. TOP_N_LABEL (20) is more than a 25 mm axis can
# show without the labels sitting on top of each other. Set to None to go back
# to TOP_N_LABEL; the file stem reverts to "_top20" when the count is 20.
DELTA_LABEL_N = 10

# Colour of the differentially correlated dots, by which side of the diagonal
# they fall on. rho_delta = rho_aged - rho_young, so a positive delta is above
# the diagonal (top left), meaning the correlation is stronger in the aged
# stratum: AGED_COL. A negative delta is below it (bottom right): YOUNG_COL.
# Set DELTA_SPLIT_COLOURS = False for the single navy colour used before.
DELTA_SPLIT_COLOURS = True
DELTA_SIG_COL = "#333399"      # used only when DELTA_SPLIT_COLOURS is False
DELTA_NS_COL = "lightgray"

SYMMETRIC_VOLCANO_X = True
VOLCANO_LABEL_FONTSIZE = 5

# Global figure settings
PLOT_DPI = 600

# ---- Small-panel figure rules ---------------------------------------------
# EXACT_FIGURE_SIZE saves at precisely the requested figsize. With
# bbox_inches="tight" matplotlib re-crops to the ink, so a 35 mm figure whose
# text overflows comes out 62 mm wide and the mm setting means nothing.
EXACT_FIGURE_SIZE = True
# A 4-entry legend measures 42 mm wide at 6 pt. The panel is 35 mm. It cannot
# go inside. A standalone legend file is written once per run instead.
VOLCANO_LEGEND = False
VOLCANO_MAX_TICKS = 3
# At 80 x 35 mm the two scatter panels have ~30 mm of width each. Per-axes
# titles and an in-axes legend do not fit alongside the axis labels.
PANEL_SUBTITLES = False
PANEL_LEGEND = False
STAT_SHORT = {"global": "depth-adj", "within_animal": "animal-adj",
              "young": "young", "aged": "aged", "delta": "aged - young"}
# ---- Rasterisation inside the SVG -----------------------------------------
# The dots were always circles. No marker= is set anywhere in this file, so
# matplotlib's default "o" is in use. They LOOK square for a different reason.
#
# Every marker here is s = 4 pt^2, i.e. 2 pt across = 0.028 inch. A rasterised
# artist inside an SVG is baked at the dpi passed to savefig, and save_figure
# passed dpi only for the PNG, so the SVG fell back to matplotlib's default of
# 100. At 100 dpi a 2 pt circle is 2.8 pixels wide, and a 3 x 3 pixel circle is
# a square. So rasterisation is the cause, but not because it changes the
# shape: the circles are simply too small to HAVE a shape at that resolution.
#
# Two independent fixes, both applied:
#   1. SVG_RASTERIZED_CORRELATION = False makes the gene correlation panels
#      true vector, so the dots are exact circles at any zoom.
#   2. SVG_RASTER_DPI is now passed when writing the SVG, so anything still
#      rasterised is baked at 600 dpi (16.7 px per marker) rather than 100.
#      Without this the volcano dots are square for exactly the same reason.
#
# The volcanoes stay rasterised because they carry one point per gene, tens of
# thousands of them. The correlation panels carry one point per indentation
# point, a couple of thousand at most. Measured on a 1,400-point panel:
# vector 237 kB, rasterised at 600 dpi 306 kB, rasterised at 100 dpi 25 kB.
# Vector is the smaller file as well as the sharper one, once the rasterised
# version is at a resolution that actually renders a circle.
SVG_RASTERIZED_SCATTER     = True     # volcanoes and other dense scatters
SVG_RASTERIZED_CORRELATION = False    # the gene correlation panels
SVG_RASTER_DPI             = 600      # dpi for whatever is still rasterised

BASE_FONT = 6
TITLE_FONT = 6
SUBTITLE_FONT = 6
AXIS_FONT = 6
TICK_FONT = 6
LEGEND_FONT = 6

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
# HELPER FUNCTIONS
# =============================================================================

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

def save_figure(fig, stem, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["png", "svg"]:
        path = out_dir / f"{stem}.{fmt}"
        kwargs = dict(bbox_inches=(None if EXACT_FIGURE_SIZE else "tight"))
        if fmt == "png":
            kwargs["dpi"] = PLOT_DPI
        else:
            # Does NOT change the SVG's dimensions, which come from figsize
            # alone. It only sets the resolution at which rasterised artists
            # are baked. Without it matplotlib falls back to 100 dpi, which is
            # what made the 2 pt markers square.
            kwargs["dpi"] = SVG_RASTER_DPI
        fig.savefig(path, format=fmt, **kwargs)
    print(f" Saved: {stem}.png / .svg")

def save_volcano_legend(out_dir, fdr_thresh=None, rho_thresh=None):
    """
    The volcano legend as its own file, placed once in the assembled figure.

    Written separately because at 6 pt the four entries measure 42 mm wide,
    wider than the 35 mm panel they would sit in.
    """
    fdr_thresh = FDR_THRESH if fdr_thresh is None else fdr_thresh
    rho_thresh = RHO_THRESH if rho_thresh is None else rho_thresh
    fig, ax = plt.subplots(figsize=(2.4, 0.9))
    ax.set_axis_off()
    h = [
        ax.scatter([], [], c="lightgrey",
                   s=VOLCANO_POINT_SIZE_NS * VOLCANO_LEGEND_SIZE_MULT,
                   label="Not significant"),
        ax.scatter([], [], c=STIFF_COL,
                   s=VOLCANO_POINT_SIZE_SIG * VOLCANO_LEGEND_SIZE_MULT,
                   label=f"FDR<{fdr_thresh}, rho>={rho_thresh:.2f}"),
        ax.scatter([], [], c=SOFT_COL,
                   s=VOLCANO_POINT_SIZE_SIG * VOLCANO_LEGEND_SIZE_MULT,
                   label=f"FDR<{fdr_thresh}, rho<=-{rho_thresh:.2f}"),
        ax.scatter([], [], facecolors="none", edgecolors="black",
                   s=MECHANOMICS_OUTLINE_SIZE * VOLCANO_LEGEND_SIZE_MULT,
                   linewidths=MECHANOMICS_OUTLINE_LW, label="Mechanomics"),
    ]
    ax.legend(handles=h, frameon=False, fontsize=LEGEND_FONT, loc="center",
              handletextpad=0.6, labelspacing=0.6)
    save_figure(fig, "0_volcano_legend", out_dir)
    plt.close(fig)

def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def clean_bool(series):
    return series.astype(str).str.strip().str.lower() == "true"

def nice_target_label(target):
    mapping = {
        "Elasticity_l_Pa": r"$E$, Young's modulus (Pa)",
        "Elasticity_eff_Pa": r"$E_{eff}$ (Pa)",
        "Loss": r"$H$, hysteresis ratio",
        "Viscosity_Ns_m": "Viscosity (Ns/m)",
    }
    return mapping.get(target, target)

def axis_target_label(target):
    lab = nice_target_label(target)
    if LOG_TARGET and target in LOG_TARGETS:
        return r"$\log_{10}$ " + lab
    return lab

COMPARTMENT_NAME_MAP = {
    "S": "S",
    "Stroma": "S",
    "StromaOSE": "S",
    "F": "F",
    "Follicle": "F",
    "CL": "CL",
    "V": "V",
    "Vasc": "V",
    "Immune": "I",
    "VascImm": "V",
    "Unknown": "Unknown",
}

def sanitize_label(text):
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")

def short_compartment_label(comp):
    comp = str(comp).strip()
    return COMPARTMENT_NAME_MAP.get(comp, sanitize_label(comp))

def make_plot_stem(plot_num, group_label, plot_type, target_col, gene=None, page=None):
    parts = [f"{plot_num}_{group_label}_{plot_type}", target_col]
    if gene is not None:
        parts.append(sanitize_label(gene))
    if page is not None:
        parts.append(f"page{page}")
    return "_".join(parts)

# =============================================================================
# REFERENCE GENE LISTS: MATRISOME, LR PAIRS, MECHANOMICS
# =============================================================================

def _read_table(path):
    """
    CSV or TSV, sniffed from the extension, with a delimiter fallback.

    encoding='utf-8-sig' matters: all three reference files start with a UTF-8
    byte-order mark, which would otherwise turn the first header into
    '\\ufeffMatrisome Division' and break every column lookup. Trailing unnamed
    all-empty columns (the matrisome masterlist has 15 of them) are dropped.
    """
    path = Path(path)
    if path.suffix.lower() in (".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
        if df.shape[1] == 1 and str(df.columns[0]).count("\t") >= 1:
            df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    junk = [c for c in df.columns
            if str(c).startswith("Unnamed:") and df[c].isna().all()]
    if junk:
        df = df.drop(columns=junk)
    return df

def _find_col(df, explicit, candidates):
    """
    Locate a column by normalised name (case, spaces, underscores and dots
    stripped). Returns None if nothing matches, so callers can decide whether
    that is fatal.
    """
    if explicit is not None:
        if explicit in df.columns:
            return explicit
        raise ValueError(f"Column '{explicit}' not in {list(df.columns)}")
    norm = {str(c).lower().replace(" ", "").replace("_", "").replace(".", ""): c
            for c in df.columns}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    return None

# Filled in by load_matrisome_table so downstream functions know which columns
# to group by. Set from the config values if those are not None.
MATRISOME_COLS = {"gene": "Gene Symbol", "category": None, "division": None}

# The combined Mechanomics annotation table, built once in main().
MECH_TABLE = None

def load_matrisome_table(matrisome_path):
    """
    Load the Naba mouse matrisome masterlist.

    Header names differ between releases: the 2012 masterlist uses
    'Division' / 'Category', some redistributions use 'Matrisome Division' /
    'Matrisome Category'. Both are detected. Whatever is found is printed and
    stored in MATRISOME_COLS, and everything downstream reads from there rather
    than assuming a spelling.
    """
    df = _read_table(matrisome_path).copy()

    gene_col = _find_col(df, MATRISOME_GENE_COL,
                         ["genesymbol", "gene", "symbol", "genesymbols",
                          "mousegenesymbol", "genename"])
    if gene_col is None:
        raise ValueError(
            f"No gene-symbol column in {Path(matrisome_path).name}. "
            f"Columns present: {list(df.columns)}. "
            f"Set MATRISOME_GENE_COL explicitly.")

    cat_col = _find_col(df, MATRISOME_CATEGORY_COL,
                        ["category", "matrisomecategory"])
    div_col = _find_col(df, MATRISOME_SUBCATEGORY_COL,
                        ["division", "matrisomedivision"])

    df[gene_col] = df[gene_col].astype(str).str.strip()
    df = df[df[gene_col].str.len() > 0]
    df = df[~df[gene_col].str.lower().isin(["nan", "none"])]

    MATRISOME_COLS["gene"] = gene_col
    MATRISOME_COLS["category"] = cat_col
    MATRISOME_COLS["division"] = div_col

    print(f" Matrisome columns: gene='{gene_col}', "
          f"category='{cat_col}', division='{div_col}'")
    if cat_col is None and div_col is None:
        print("   [WARN] no category/division column found, so the Fisher test "
              "will only report 'all matrisome'.")
    for c in (div_col, cat_col):
        if c is not None:
            print(f"   {c}: {df[c].astype(str).nunique()} level(s) -> "
                  f"{sorted(df[c].astype(str).unique())[:8]}")
    return df

def _clean_symbol(v):
    """
    Pull the primary gene symbol out of a connectomeDB symbol cell.

    Cells look like 'Lrp1 (CD91, A2mr, b2b1554Clo)'. 34 receptor cells are also
    corrupted with a stray HTML anchor fragment, e.g.
    ', Cd, ska26, skax26)">Lrp6 (ska, Cd, ska26, skax26)', where the real symbol
    only appears after '">'. Both are handled here.
    """
    v = str(v)
    if '">' in v:
        v = v.split('">')[-1]
    v = v.split("(")[0]
    return v.strip().strip(",").strip()

def load_lr_pairs(lr_path):
    """
    connectomeDB ligand-receptor pairs.

    The 'LR Pair' column ('Apob Lrp6') is the authoritative parse: it is clean
    in all 3,252 rows and agrees exactly with the parsed symbol columns, which
    the HTML corruption above does not. Symbol columns are the fallback.

    Returns a DataFrame with ligand, receptor and the two location columns, or
    None if the file is missing or unreadable.
    """
    if not Path(lr_path).exists():
        print(f" WARNING: LR pair table not found at {lr_path}")
        return None
    try:
        df = _read_table(lr_path)
    except Exception as e:
        print(f" WARNING: could not read LR pair table: {e}")
        return None

    pair_col = _find_col(df, None, ["lrpair", "pair", "ligandreceptorpair"])
    lig = _find_col(df, LR_LIGAND_COL,
                    ["ligandsymbols", "ligandgenesymbol", "ligandsymbol",
                     "ligand", "ligandgene", "source", "from", "lig"])
    rec = _find_col(df, LR_RECEPTOR_COL,
                    ["receptorsymbols", "receptorgenesymbol", "receptorsymbol",
                     "receptor", "receptorgene", "target", "to", "rec"])

    used = None
    if pair_col is not None:
        sp = df[pair_col].astype(str).str.strip().str.split(r"\s+", n=1,
                                                            expand=True)
        if sp.shape[1] == 2 and sp[1].notna().all():
            out = pd.DataFrame({"ligand": sp[0].str.strip(),
                                "receptor": sp[1].str.strip()})
            used = f"'{pair_col}' (split on whitespace)"

    if used is None:
        if lig is None or rec is None:
            print(f" WARNING: could not identify ligand/receptor columns in "
                  f"{Path(lr_path).name}. Columns: {list(df.columns)}")
            print("          Set LR_LIGAND_COL and LR_RECEPTOR_COL explicitly.")
            return None
        out = pd.DataFrame({"ligand": df[lig].map(_clean_symbol),
                            "receptor": df[rec].map(_clean_symbol)})
        used = f"'{lig}' / '{rec}'"

    lig_loc = _find_col(df, None, ["ligandlocation"])
    rec_loc = _find_col(df, None, ["receptorlocation"])
    out["ligand_location"] = df[lig_loc].astype(str) if lig_loc else ""
    out["receptor_location"] = df[rec_loc].astype(str) if rec_loc else ""

    for c in ("ligand", "receptor"):
        out[c] = out[c].astype(str).str.strip()
    out = out[(out["ligand"] != "") & (out["receptor"] != "")]
    out = out[~out["ligand"].str.lower().isin(["nan", "none"])]
    out = out[~out["receptor"].str.lower().isin(["nan", "none"])]
    out = out.drop_duplicates(subset=["ligand", "receptor"]).reset_index(drop=True)

    # Cross-check the two parses when both are available, so a future file
    # format change is loud rather than silent.
    if pair_col is not None and lig is not None and rec is not None:
        alt_l = df[lig].map(_clean_symbol).str.strip()
        alt_r = df[rec].map(_clean_symbol).str.strip()
        both = pd.DataFrame({"ligand": df[pair_col].astype(str).str.strip()
                             .str.split(r"\s+", n=1, expand=True)[0].str.strip(),
                             "receptor": df[pair_col].astype(str).str.strip()
                             .str.split(r"\s+", n=1, expand=True)[1].str.strip()})
        agree = float(((both["ligand"] == alt_l) & (both["receptor"] == alt_r)).mean())
        print(f" LR parse cross-check: {agree:.3f} agreement between "
              f"'{pair_col}' and the symbol columns")
        if agree < 0.95:
            print("   [WARN] the two parses disagree; inspect the file before "
                  "trusting the LR annotation.")

    print(f" LR pair table: {len(out):,} unique pairs from {used}; "
          f"{out['ligand'].nunique():,} ligands, "
          f"{out['receptor'].nunique():,} receptors")
    return out

def load_goi_table(goi_path):
    """
    Custom genes-of-interest table. Returns a DataFrame with a 'gene' column
    plus whatever annotation columns exist, or None if the file is missing.
    """
    if not Path(goi_path).exists():
        print(f" WARNING: GOI list not found at {goi_path}; "
              f"falling back to the list hardcoded in this script.")
        return pd.DataFrame({"gene": list(GENES_OF_INTEREST_FALLBACK)})
    try:
        df = _read_table(goi_path)
    except Exception as e:
        print(f" WARNING: could not read GOI list ({e}); using the hardcoded list.")
        return pd.DataFrame({"gene": list(GENES_OF_INTEREST_FALLBACK)})

    col = _find_col(df, GOI_GENE_COL,
                    ["genes", "gene", "genesymbol", "symbol", "geneid",
                     "genename", "goi"])
    if col is None:
        col = df.columns[0]
        print(f" GOI list: no obvious gene column, using the first column "
              f"'{col}'. Columns present: {list(df.columns)}")

    out = df.rename(columns={col: "gene"}).copy()
    out["gene"] = out["gene"].astype(str).str.strip()
    out = out[(out["gene"] != "") & (~out["gene"].str.lower().isin(["nan", "none"]))]

    n_raw, n_uniq = len(out), out["gene"].nunique()
    print(f" GOI list: {n_uniq:,} unique symbols from column '{col}' "
          f"of {Path(goi_path).name}"
          + (f" ({n_raw - n_uniq} duplicate row(s) merged)" if n_raw > n_uniq else ""))

    for old, new in (("Ctgf", "Ccn2"), ("Cyr61", "Ccn1")):
        if old in set(out["gene"]):
            print(f"   NOTE: '{old}' is a retired symbol. The current one is "
                  f"'{new}', which is "
                  f"{'already' if new in set(out['gene']) else 'NOT'} in your list.")
    return out

def _join_unique(values, sep="; ", limit=None):
    seen = []
    for v in values:
        v = str(v).strip()
        if v and v.lower() not in ("nan", "none", "-") and v not in seen:
            seen.append(v)
    if limit is not None and len(seen) > limit:
        seen = seen[:limit] + [f"...(+{len(values) - limit})"]
    return sep.join(seen)

def build_mechanomics_table(matrisome_df, lr_df, goi_df, universe=None):
    """
    One table, one row per gene.

    MEMBERSHIP
        M  matrisome masterlist
        C  connectomeDB (as a ligand and/or a receptor)
        G  custom genes-of-interest list

    The Mechanomics set is M UNION G. A gene that is ONLY in connectomeDB is
    deliberately excluded: connectomeDB is used to annotate ligand-receptor
    relationships among genes you already care about, not to import ~873 extra
    genes you never selected. So 'source' can be M, G, MC, MG, CG or MCG, but
    never C alone. Set MECHANOMICS_ADD_LR_PARTNERS = True to change that.

    COLUMNS
        gene                  symbol
        source                membership code, letters in the order M, C, G
        in_matrisome / in_connectome / in_goi
        connectome_role       ligand, receptor, ligand+receptor, or NA
        receptor_of           ligands for which this gene is the receptor
        ligand_of             receptors that this gene, as a ligand, binds
        receptor_of_in_set    the same, restricted to Mechanomics members
        ligand_of_in_set      the same, restricted to Mechanomics members
        n_lr_partners         count of distinct partners
        info                  everything else from the three files, compressed
        in_expression_matrix  present as a column of the expression matrix
    """
    from collections import defaultdict

    # ---------------------------------------------------------------- sources
    M, mat_info = set(), {}
    if matrisome_df is not None:
        gcol = MATRISOME_COLS["gene"]
        dcol, ccol = MATRISOME_COLS["division"], MATRISOME_COLS["category"]
        ncol = _find_col(matrisome_df, None, ["genename"])
        scol = _find_col(matrisome_df, None, ["synonyms"])
        for _, r in matrisome_df.iterrows():
            g = str(r[gcol]).strip()
            if not g:
                continue
            M.add(g)
            bits = [str(r[c]).strip() for c in (dcol, ccol, ncol)
                    if c is not None and pd.notna(r.get(c))]
            syn = str(r[scol]).strip() if scol is not None and pd.notna(r.get(scol)) else ""
            if syn and syn != "-":
                bits.append(f"syn={syn}")
            mat_info[g] = " | ".join(b for b in bits if b and b.lower() != "nan")

    G, goi_info = set(), {}
    if goi_df is not None and len(goi_df):
        jcol = _find_col(goi_df, None, ["justification"])
        kcol = _find_col(goi_df, None, ["category"])
        rcol = _find_col(goi_df, None, ["ref", "reference", "source"])
        for g, sub in goi_df.groupby("gene"):
            g = str(g).strip()
            if not g:
                continue
            G.add(g)
            bits = []
            for c in (kcol, jcol, rcol):
                if c is not None:
                    j = _join_unique(sub[c].tolist())
                    if j:
                        bits.append(j)
            goi_info[g] = " | ".join(bits)

    C = set()
    receptor_of = defaultdict(set)   # gene -> ligands it receives
    ligand_of = defaultdict(set)     # gene -> receptors it binds
    loc = defaultdict(set)
    if lr_df is not None and len(lr_df):
        for lg, rc, ll, rl in zip(lr_df["ligand"], lr_df["receptor"],
                                  lr_df["ligand_location"],
                                  lr_df["receptor_location"]):
            C.add(lg)
            C.add(rc)
            ligand_of[lg].add(rc)
            receptor_of[rc].add(lg)
            if str(ll).strip() and str(ll).lower() != "nan":
                loc[lg].add(f"ligand loc={str(ll).strip()}")
            if str(rl).strip() and str(rl).lower() != "nan":
                loc[rc].add(f"receptor loc={str(rl).strip()}")

    # ---------------------------------------------------------------- the set
    mech = M | G
    if MECHANOMICS_ADD_LR_PARTNERS and lr_df is not None:
        partners = set()
        for g in list(mech):
            partners |= ligand_of.get(g, set()) | receptor_of.get(g, set())
        added = partners - mech
        mech |= partners
        print(f"   MECHANOMICS_ADD_LR_PARTNERS is True: {len(added):,} "
              f"connectome-only partner(s) added, coded 'C'.")

    # ---------------------------------------------------------------- rows
    rows = []
    for g in sorted(mech):
        in_m, in_c, in_g = g in M, g in C, g in G
        src = ("M" if in_m else "") + ("C" if in_c else "") + ("G" if in_g else "")
        is_lig, is_rec = g in ligand_of, g in receptor_of
        role = ("ligand+receptor" if is_lig and is_rec else
                "ligand" if is_lig else "receptor" if is_rec else np.nan)

        r_of = sorted(receptor_of.get(g, set()))
        l_of = sorted(ligand_of.get(g, set()))
        r_in = [x for x in r_of if x in mech]
        l_in = [x for x in l_of if x in mech]

        info_bits = []
        if in_m and mat_info.get(g):
            info_bits.append("M: " + mat_info[g])
        if in_g and goi_info.get(g):
            info_bits.append("G: " + goi_info[g])
        if in_c and loc.get(g):
            info_bits.append("C: " + _join_unique(sorted(loc[g])))

        rows.append({
            "gene": g,
            "source": src,
            "in_matrisome": in_m,
            "in_connectome": in_c,
            "in_goi": in_g,
            "connectome_role": role,
            "receptor_of": "; ".join(r_of) if r_of else np.nan,
            "ligand_of": "; ".join(l_of) if l_of else np.nan,
            "receptor_of_in_set": "; ".join(r_in) if r_in else np.nan,
            "ligand_of_in_set": "; ".join(l_in) if l_in else np.nan,
            "n_lr_partners": len(set(r_of) | set(l_of)),
            "info": " || ".join(info_bits) if info_bits else np.nan,
        })

    tab = pd.DataFrame(rows)

    print("\n Mechanomics gene set  (M = matrisome, C = connectomeDB, "
          "G = genes of interest)")
    print(f"   matrisome              : {len(M):,}")
    print(f"   genes of interest      : {len(G):,}")
    print(f"   connectomeDB           : {len(C):,} "
          f"({len(C - M - G):,} of them are connectome-only and are "
          f"{'INCLUDED' if MECHANOMICS_ADD_LR_PARTNERS else 'EXCLUDED'})")
    print(f"   MECHANOMICS TOTAL      : {len(tab):,}")
    if len(tab):
        vc = tab["source"].value_counts()
        for code in ["MCG", "MC", "MG", "CG", "M", "G", "C"]:
            if code in vc.index:
                print(f"     {code:<4s} {int(vc[code]):>6,}")
        n_lr = int((tab["n_lr_partners"] > 0).sum())
        print(f"   with an LR partner     : {n_lr:,}")

    if universe is not None:
        uni = set(universe)
        tab["in_expression_matrix"] = tab["gene"].isin(uni)
        usable = set(tab.loc[tab["in_expression_matrix"], "gene"])
        print(f"   present in expression  : {len(usable):,}")
        return tab, usable
    tab["in_expression_matrix"] = np.nan
    return tab, set(tab["gene"])

# =============================================================================
# STEP 1: LOAD EXPRESSION MATRIX AND METADATA
# =============================================================================

def load_expression_counts(expr_dir):
    """
    Load the expression matrix as RAW COUNTS, bins x genes.

    If step9_counts_matrix.mtx exists it is used directly. Otherwise the
    log-normalised matrix is inverted exactly:

        counts = expm1(data) * nCount_RNA / scale_factor

    Seurat's LogNormalize is log1p(counts / nCount_RNA * scale.factor), so this
    is its exact inverse up to the scale factor, which cancels when the result
    is rounded to the smallest observed unit. The integer check below is what
    proves the inversion worked; if it fails, re-export counts from R rather
    than trusting the reconstruction.
    """
    print(f"Loading expression matrix from: {expr_dir}")

    with open(expr_dir / EXPR_BARCODES_FILE, "r", encoding="utf-8") as f:
        barcodes = [l.strip() for l in f]
    with open(expr_dir / EXPR_FEATURES_FILE, "r", encoding="utf-8") as f:
        genes = [l.strip() for l in f]

    counts_path = expr_dir / EXPR_COUNTS_FILE
    if counts_path.exists():
        print(f" Using raw counts export: {counts_path.name}")
        X = sio.mmread(counts_path).tocsr()
        is_counts = True
    else:
        print(f" Raw counts export not found; loading {EXPR_MATRIX_FILE} "
              f"and inverting LogNormalize")
        X = sio.mmread(expr_dir / EXPR_MATRIX_FILE).tocsr()
        is_counts = False

    # R export is genes x cells, so transpose to cells x genes
    if X.shape[0] == len(genes) and X.shape[1] == len(barcodes):
        X = X.T.tocsr()

    print(f" Expression matrix: {X.shape[0]:,} bins x {X.shape[1]:,} genes")

    if X.shape[0] != len(barcodes):
        raise ValueError(
            f"Matrix rows ({X.shape[0]}) do not match number of barcodes "
            f"({len(barcodes)}).")
    if X.shape[1] != len(genes):
        raise ValueError(
            f"Matrix columns ({X.shape[1]}) do not match number of genes "
            f"({len(genes)}).")

    return X, barcodes, genes, is_counts

def reconstruct_counts(X, depth_per_bin):
    """Invert Seurat LogNormalize. Raises if the result is not integral."""
    print(" Reconstructing counts from log-normalised values ...")
    nC = np.asarray(depth_per_bin, dtype=float)
    if np.isnan(nC).any():
        raise ValueError(
            f"{DEPTH_COL} has {int(np.isnan(nC).sum()):,} missing values; "
            f"counts cannot be reconstructed.")
    R = sparse.csr_matrix(X, copy=True)
    R.data = np.expm1(R.data)
    R.data *= np.repeat(nC, np.diff(R.indptr))
    R.eliminate_zeros()
    if R.data.size == 0:
        raise ValueError("Expression matrix is empty after reconstruction.")
    unit = float(R.data.min())
    ratio = R.data / unit
    ok = float((np.abs(ratio - np.round(ratio)) < 1e-4).mean())
    print(f"   integer check: {ok:.5f} of non-zero entries are integer multiples")
    if ok < 0.99:
        raise ValueError(
            "Counts reconstruction failed the integer check. Re-export raw "
            "counts from R as step9_counts_matrix.mtx and re-run.")
    R.data = np.round(ratio)
    R.eliminate_zeros()
    return R.tocsr()

def build_barcode_full(df):
    if "barcode_full" in df.columns:
        return df
    if "Unique Barcodes" in df.columns:
        df["barcode_full"] = df["Unique Barcodes"].astype(str).str.strip()
        return df
    if "Barcodes" in df.columns and "samples" in df.columns:
        df["barcode_full"] = (
            df["Barcodes"].astype(str).str.strip() + "." +
            df["samples"].astype(str).str.strip()
        )
        return df
    return df

def load_matched_metadata(expr_dir, metadata_tsv, target_col):
    matched_path = expr_dir / EXPR_MATCHED_METADATA_FILE

    if matched_path.exists():
        print(f"\nLoading matched metadata export: {matched_path}")
        meta = pd.read_csv(matched_path, sep="\t", low_memory=False)
    else:
        print(f"\nMatched metadata file not found, falling back to full metadata TSV:")
        print(f" {metadata_tsv}")
        meta = pd.read_csv(metadata_tsv, sep="\t", low_memory=False)

    print(f" Total rows loaded: {len(meta):,}")

    meta = meta.reset_index(drop=True).copy()
    meta["expr_row"] = np.arange(len(meta))
    meta = build_barcode_full(meta)

    required_cols = [
        target_col, NANO_ID_COL, AGE_COL, CLUSTER_COL, TISSUE_COL,
        COMPARTMENT_COL, ROI_COL, COL_X, COL_Y,
    ]
    missing_cols = [c for c in required_cols if c not in meta.columns]

    if missing_cols:
        print(f" Matched metadata is missing columns: {missing_cols}")
        print(f" Loading full metadata TSV to recover them: {metadata_tsv}")

        meta_full = pd.read_csv(metadata_tsv, sep="\t", low_memory=False)
        meta_full = build_barcode_full(meta_full)

        if "barcode_full" not in meta.columns:
            raise ValueError(
                "Matched metadata is missing required columns and has no "
                "'barcode_full' to join on.")
        if "barcode_full" not in meta_full.columns:
            raise ValueError(
                "Full metadata TSV has no 'barcode_full' and it could not be "
                "reconstructed from Barcodes + samples.")

        recoverable = ["barcode_full"] + [
            c for c in required_cols
            if c not in meta.columns and c in meta_full.columns
        ]
        still_missing = [c for c in required_cols
                         if c not in meta.columns and c not in meta_full.columns]
        if still_missing:
            raise ValueError(
                f"Missing from both matched and full metadata: {still_missing}")

        meta = meta.merge(
            meta_full[recoverable].drop_duplicates(subset=["barcode_full"]),
            on="barcode_full", how="left")

        missing_cols = [c for c in required_cols if c not in meta.columns]
        if missing_cols:
            raise ValueError(
                f"Still missing after recovery: {missing_cols}")
        print(" Successfully recovered missing columns from full metadata TSV.")

    meta[target_col] = pd.to_numeric(meta[target_col], errors="coerce")
    keep_mask = meta[target_col].notna()
    if "valid_measurement" in meta.columns:
        keep_mask &= clean_bool(meta["valid_measurement"])
    keep_mask &= meta[NANO_ID_COL].notna()
    keep_mask &= meta[NANO_ID_COL].astype(str).str.strip().ne("N/A")

    meta = meta.loc[keep_mask].copy()
    meta["age_group"] = meta[AGE_COL].apply(
        lambda s: "young" if str(s).strip().lower() == "young" else "aged")

    # Animal identity. Tissue.slices is like "YF1-2"; the trailing -N is the
    # section, so stripping it gives the animal.
    meta["animal"] = (meta[TISSUE_COL].astype(str)
                      .str.replace(r"-\d+$", "", regex=True))

    meta["compartment_short"] = meta[COMPARTMENT_COL].apply(short_compartment_label)

    print(f" 4 um bins with valid {target_col}: {len(meta):,}")
    print(f" Age groups (bins): {meta['age_group'].value_counts().to_dict()}")
    print(f" Animals: {sorted(meta['animal'].unique().tolist())}")

    return meta

# =============================================================================
# STEP 2: AGGREGATE 4 um BINS TO 50 um NANOINDENTATION POINTS
# =============================================================================

def build_point_index(meta):
    """
    Composite point key -> integer index per bin, plus the point-level table.

    Returns
    -------
    inv         int array, one entry per bin, giving its point index
    point_meta  DataFrame, one row per point
    """
    key_cols = [c for c in POINT_KEY_COLS if c in meta.columns]
    if not key_cols:
        raise ValueError(f"None of {POINT_KEY_COLS} present in metadata.")
    pid = meta[key_cols].astype(str).agg("||".join, axis=1).values
    uniq, inv = np.unique(pid, return_inverse=True)

    print(f"\n Point key: {' + '.join(key_cols)}")
    print(f"   {len(uniq):,} points "
          f"(vs {meta[NANO_ID_COL].nunique():,} on {NANO_ID_COL} alone)")
    return uniq, inv, key_cols

def majority_by_point(values, inv, n_points, default="Unknown"):
    s = pd.Series(np.asarray(values))
    out = (s.groupby(inv)
           .agg(lambda v: v.dropna().astype(str).value_counts().idxmax()
                if v.notna().any() else default)
           .reindex(range(n_points)).values)
    return np.where(pd.isna(out), default, out)

def aggregate_expression_to_points(counts, meta, inv, n_points, target_col,
                                   bin_mask=None):
    """
    Sum raw counts over the bins of each nanoindentation point, then normalise
    ONCE per point. This is the whole reason the depth artefact goes away: the
    old code normalised per bin and then conditioned on expression, which made
    every gene a proxy for 1/nCount.

    bin_mask, if given, restricts aggregation to a subset of bins (used for the
    compartment-specific analyses, where only that compartment's bins under a
    point contribute).

    Returns
    -------
    lognorm   csc_matrix, n_points x n_genes, log1p(CPM-like)
    raw_sum   csr_matrix, n_points x n_genes, summed counts (for gene filters)
    pmeta     DataFrame, n_points rows
    """
    n_bins = counts.shape[0]
    if len(meta) != n_bins:
        raise ValueError(
            f"Metadata rows ({len(meta)}) do not match expression rows "
            f"({n_bins}).")

    if bin_mask is None:
        bin_mask = np.ones(n_bins, dtype=bool)
    bin_mask = np.asarray(bin_mask, dtype=bool)

    rows = inv[bin_mask]
    cols = np.nonzero(bin_mask)[0]
    indicator = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n_points, n_bins))

    raw_sum = (indicator @ counts).tocsr()
    n_bins_used = np.asarray(indicator.sum(axis=1)).ravel()

    depth = np.asarray(raw_sum.sum(axis=1)).ravel()
    scale = np.where(depth > 0, SCALE_FACTOR / np.maximum(depth, 1.0), 0.0)
    lognorm = sparse.csr_matrix(raw_sum.multiply(scale[:, None]))
    lognorm.data = np.log1p(lognorm.data)
    lognorm = lognorm.tocsc()

    # Point-level metadata. Target, age, animal, tissue, ROI are constant within
    # a point by construction, so "first" is exact; they are checked below.
    pmeta = pd.DataFrame(index=np.arange(n_points))
    pmeta["n_bins_used"] = n_bins_used
    pmeta["depth"] = depth
    pmeta["log_depth"] = np.log10(np.maximum(depth, 1.0))

    tgt = pd.to_numeric(meta[target_col], errors="coerce").values
    with np.errstate(invalid="ignore"):
        tsum = np.bincount(inv, weights=np.nan_to_num(tgt), minlength=n_points)
        tcnt = np.bincount(inv, weights=(~np.isnan(tgt)).astype(float),
                           minlength=n_points)
        tmax = pd.Series(tgt).groupby(inv).max().reindex(range(n_points)).values
        tmin = pd.Series(tgt).groupby(inv).min().reindex(range(n_points)).values
    pmeta[target_col] = np.where(tcnt > 0, tsum / np.maximum(tcnt, 1), np.nan)
    spread = np.abs(np.asarray(tmax, float) - np.asarray(tmin, float))
    n_multi = int(np.nansum(spread > 1e-9))
    if n_multi:
        print(f"   [WARN] {n_multi:,} point(s) hold more than one distinct "
              f"{target_col} value. The point key is not unique enough.")

    for col, name in [(AGE_COL, "age_raw"), (TISSUE_COL, "tissue"),
                      (ROI_COL, "roi"), ("animal", "animal"),
                      (CLUSTER_COL, "majority_cluster"),
                      (COMPARTMENT_COL, "majority_compartment")]:
        if col in meta.columns:
            pmeta[name] = majority_by_point(meta[col].values, inv, n_points)

    if "age_raw" in pmeta.columns:
        pmeta["age_group"] = np.where(
            pmeta["age_raw"].astype(str).str.strip().str.lower() == "young",
            "young", "aged")
    else:
        pmeta["age_group"] = "aged"

    for col, name in [(COL_X, "x_mean"), (COL_Y, "y_mean")]:
        if col in meta.columns:
            v = pd.to_numeric(meta[col], errors="coerce").values
            pmeta[name] = (np.bincount(inv, weights=np.nan_to_num(v),
                                       minlength=n_points)
                           / np.maximum(np.bincount(inv, minlength=n_points), 1))

    for col, name in [(GRID_X_COL, "grid_x"), (GRID_Y_COL, "grid_y")]:
        if col in meta.columns:
            pmeta[name] = majority_by_point(meta[col].values, inv, n_points)

    return lognorm, raw_sum, pmeta

def compartment_fractions(meta, inv, n_points):
    """Fraction of each point's 4 um bins falling in each compartment."""
    comp = meta["compartment_short"].astype(str).values
    tab = pd.crosstab(pd.Series(inv, name="point"), pd.Series(comp, name="comp"))
    tab = tab.reindex(range(n_points), fill_value=0)
    frac = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return tab, frac

# =============================================================================
# STEP 3: PARTIAL SPEARMAN MACHINERY
# =============================================================================

def rank_resid_basis(covars):
    """
    Design matrix for rank residualisation.

    Partial Spearman is Pearson correlation of the residuals after regressing
    the RANKS of x and y on the RANKS of the covariates. Indicator columns
    (animal dummies, age) are already 0/1 so ranking them is a monotone
    relabelling and changes nothing.
    """
    C = np.asarray(covars, float)
    if C.ndim == 1:
        C = C[:, None]
    n = C.shape[0]
    R = np.column_stack([rankdata(C[:, j]).astype(float)
                         for j in range(C.shape[1])])
    Z = np.column_stack([np.ones(n), R])
    return Z, np.linalg.pinv(Z)

def prep_target(y, covars):
    """Pre-residualise the target once; every gene reuses it."""
    Z, P = rank_resid_basis(covars)
    ry = rankdata(y).astype(float)
    ey = ry - Z @ (P @ ry)
    ny = float(np.sqrt(np.dot(ey, ey)))
    return dict(Z=Z, P=P, ey=ey, ny=ny, k=Z.shape[1] - 1, n=len(y))

def partial_residuals(x, model):
    """Rank residuals of x after the covariates. Used by the scatter panels."""
    rx = rankdata(np.asarray(x, float)).astype(float)
    return rx - model["Z"] @ (model["P"] @ rx)

def partial_rho(x, model):
    """
    Partial Spearman of x against the model's pre-residualised target.

    The degenerate-input guard matters: for a constant x the residuals are
    ~1e-11 rather than exactly 0, so an `nx == 0` test never fires and the
    function returns a meaningless rho of ~0.02. The guard is relative to the
    centred norm of the ranks.
    """
    x = np.asarray(x, float)
    if x.size == 0 or not np.isfinite(x).all():
        return np.nan, np.nan
    if float(x.max() - x.min()) == 0.0:
        return np.nan, np.nan
    rx = rankdata(x).astype(float)
    ex = rx - model["Z"] @ (model["P"] @ rx)
    nx = float(np.sqrt(np.dot(ex, ex)))
    cx = float(np.sqrt(np.dot(rx - rx.mean(), rx - rx.mean())))
    if model["ny"] <= 0 or nx <= 1e-8 * max(cx, 1e-12):
        return np.nan, np.nan
    r = float(np.dot(ex, model["ey"]) / (nx * model["ny"]))
    r = float(np.clip(r, -1.0, 1.0))
    df = model["n"] - 2 - model["k"]
    if df <= 0 or abs(r) >= 1:
        return r, np.nan
    tt = r * np.sqrt(df / (1.0 - r ** 2))
    return r, float(2 * student_t.sf(abs(tt), df))

def spearman_diff_test(r1, n1, r2, n2):
    """
    Fisher z test for two independent Spearman correlations. The 1.06 factor is
    the standard variance inflation for Spearman relative to Pearson.
    """
    if not (np.isfinite(r1) and np.isfinite(r2)):
        return np.nan
    if n1 < 6 or n2 < 6 or abs(r1) >= 1 or abs(r2) >= 1:
        return np.nan
    se = np.sqrt(1.06 / (n1 - 3) + 1.06 / (n2 - 3))
    z = (np.arctanh(r1) - np.arctanh(r2)) / se
    return float(2 * normal_dist.sf(abs(z)))

def animal_dummies(animals, mask=None):
    """Reference-coded indicator columns, or None if fewer than two animals."""
    a = np.asarray(animals)
    if mask is not None:
        a = a[mask]
    levels = np.unique(a)
    if len(levels) < 2:
        return None
    return np.column_stack([(a == lv).astype(float) for lv in levels[1:]])

def build_models(y, log_depth, animals, is_young, n_points):
    """
    The five statistics, as covariate sets.

    global         depth, plus age if AGE_AS_COVARIATE
    within_animal  depth + animal dummies. Animal is nested in age, so this
                   also removes the age effect, and any residual signal is
                   within-animal variation in stiffness.
    young / aged   stratum only, depth + animal within that stratum
    """
    models = {}
    is_aged = ~is_young
    age_num = is_aged.astype(float)

    if AGE_AS_COVARIATE and is_young.any() and is_aged.any():
        cov_global = np.column_stack([log_depth, age_num])
    else:
        cov_global = log_depth[:, None]
    models["global"] = prep_target(y, cov_global)

    dA = animal_dummies(animals)
    models["within_animal"] = prep_target(
        y, np.column_stack([log_depth, dA]) if dA is not None
        else log_depth[:, None])

    for name, msk in (("young", is_young), ("aged", is_aged)):
        if int(msk.sum()) >= MIN_POINTS_PER_STRATUM:
            d_ = animal_dummies(animals, msk)
            models[name] = prep_target(
                y[msk],
                np.column_stack([log_depth[msk], d_]) if d_ is not None
                else log_depth[msk][:, None])
    return models

def expression_subset_mask(raw_v):
    """Per-gene point mask used consistently in stats and plotting."""
    raw_v = np.asarray(raw_v).ravel()
    if EXPRESSION_SUBSET_MODE == "raw_count":
        return raw_v >= EXPRESSION_SUBSET_MIN_COUNT
    raise ValueError(f"Unknown EXPRESSION_SUBSET_MODE: {EXPRESSION_SUBSET_MODE}")

def compute_point_correlations(lognorm, raw_sum, gene_names, y, log_depth,
                               animals, is_young, detect_frac, total_counts):
    """
    One pass over genes.

    Genes are screened upstream on all retained points, but the correlation for
    each gene is computed only on the points where that gene is detected above
    EXPRESSION_SUBSET_MIN_COUNT. The covariate-adjustment models are rebuilt on
    that same subset so x, y and the nuisance terms are aligned.
    """
    rows = []
    Xc = lognorm.tocsc()
    Rc = raw_sum.tocsc()
    for j, g in enumerate(gene_names):
        v = np.asarray(Xc[:, j].todense()).ravel()
        raw_v = np.asarray(Rc[:, j].todense()).ravel()
        expr_mask = expression_subset_mask(raw_v)

        n_expr = int(expr_mask.sum())
        n_y = int((expr_mask & is_young).sum())
        n_a = int((expr_mask & (~is_young)).sum())

        rec = {"gene": g,
               "detect_frac": float(detect_frac[j]),
               "total_counts": float(total_counts[j]),
               "n_points_expr": n_expr,
               "n_points_expr_young": n_y,
               "n_points_expr_aged": n_a}

        if n_expr < MIN_POINTS_PER_GENE_CORR:
            for name in ("global", "within_animal", "young", "aged"):
                rec[f"rho_{name}"] = np.nan
                rec[f"p_{name}"] = np.nan
            rec["rho_delta"] = np.nan
            rec["p_delta"] = np.nan
            rows.append(rec)
            continue

        y_sub = y[expr_mask]
        log_depth_sub = log_depth[expr_mask]
        animals_sub = animals[expr_mask]
        is_young_sub = is_young[expr_mask]
        models = build_models(y_sub, log_depth_sub, animals_sub,
                              is_young_sub, n_expr)

        for name in ("global", "within_animal"):
            r, p = partial_rho(v[expr_mask], models[name])
            rec[f"rho_{name}"], rec[f"p_{name}"] = r, p

        is_aged_sub = ~is_young_sub
        for name, msk in (("young", is_young_sub), ("aged", is_aged_sub)):
            if name in models:
                r, p = partial_rho(v[expr_mask][msk], models[name])
            else:
                r, p = np.nan, np.nan
            rec[f"rho_{name}"], rec[f"p_{name}"] = r, p

        ra, ry = rec.get("rho_aged", np.nan), rec.get("rho_young", np.nan)
        rec["rho_delta"] = (ra - ry) if (np.isfinite(ra) and np.isfinite(ry)) \
            else np.nan
        rec["p_delta"] = spearman_diff_test(ra, n_a, ry, n_y)
        rows.append(rec)

    res = pd.DataFrame(rows)
    for name in ("global", "within_animal", "young", "aged", "delta"):
        res[f"fdr_{name}"] = np.nan
        ok = res[f"p_{name}"].notna()
        if int(ok.sum()):
            res.loc[ok, f"fdr_{name}"] = multipletests(
                res.loc[ok, f"p_{name}"].values, method="fdr_bh")[1]
    return res

# =============================================================================
# STEP 4: VOLCANO
# =============================================================================

def select_top_genes(sig_pos_df, sig_neg_df, n_total, rho_pos="rho",
                     rho_neg="rho"):
    """
    Balanced pick: half from each direction, and if one direction is short the
    other fills the gap. Concatenating the two frames and slicing the head is
    the bug that once gave 40 negatives and zero positives in every panel.
    """
    half = max(n_total // 2, 1)
    pos = sig_pos_df.head(half)["gene"].tolist()
    neg = sig_neg_df.head(half)["gene"].tolist()
    out = []
    for a, b in zip(pos + [None] * len(neg), neg + [None] * len(pos)):
        for g in (a, b):
            if g is not None and g not in out:
                out.append(g)
    if len(out) < n_total:
        for g in (sig_pos_df["gene"].tolist() + sig_neg_df["gene"].tolist()):
            if g not in out:
                out.append(g)
            if len(out) >= n_total:
                break
    return out[:n_total]

def plot_volcano_correlations(res, stat, out_dir, target_col,
                              group_label="global", mechanomics=None,
                              rho_thresh=RHO_THRESH, fdr_thresh=FDR_THRESH,
                              n_points=None, subtitle=""):
    """
    Volcano of -log10(FDR) against partial rho for one statistic.

    Dots are half the size used at bin level. Genes in the Mechanomics set that
    are also significant get a black outline; only the top MECHANOMICS_LABEL_N
    positive and negative of THOSE are labelled, so the figure stays readable.
    """
    rho_col, fdr_col, p_col = f"rho_{stat}", f"fdr_{stat}", f"p_{stat}"
    if rho_col not in res.columns:
        print(f" No column {rho_col}; skipping volcano.")
        return None, None, None, None, []

    d = res[["gene", rho_col, p_col, fdr_col]].dropna(
        subset=["gene", rho_col, fdr_col]).copy()
    if d.empty:
        print(f" {group_label} / {stat}: nothing to plot.")
        return None, None, None, None, []

    d = d.rename(columns={rho_col: "rho", p_col: "pval", fdr_col: "fdr"})
    d["neglog10_fdr"] = -np.log10(d["fdr"].clip(lower=1e-300))
    d["direction"] = np.where(d["rho"] > 0, "positive",
                              np.where(d["rho"] < 0, "negative", "zero"))
    d = d.sort_values("fdr")

    sig_pos_mask = (d["fdr"] < fdr_thresh) & (d["rho"] >= rho_thresh)
    sig_neg_mask = (d["fdr"] < fdr_thresh) & (d["rho"] <= -rho_thresh)
    nonsig_mask = ~(sig_pos_mask | sig_neg_mask)

    sig_pos_df = d[sig_pos_mask].sort_values("rho", ascending=False).copy()
    sig_neg_df = d[sig_neg_mask].sort_values("rho", ascending=True).copy()
    sig_combined = pd.concat([sig_neg_df, sig_pos_df], ignore_index=True)
    sig_genes = sig_combined["gene"].astype(str).tolist()

    stem_base = f"{group_label}_{target_col}_{stat_disp(stat)}"
    d.to_csv(out_dir / f"{stem_base}_volcano_table_all_genes.csv", index=False)
    sig_pos_df.to_csv(out_dir / f"{stem_base}_significant_positive_genes.csv",
                      index=False)
    sig_neg_df.to_csv(out_dir / f"{stem_base}_significant_negative_genes.csv",
                      index=False)
    sig_combined.to_csv(
        out_dir / f"{stem_base}_significant_correlated_genes_combined.csv",
        index=False)

    npts = f", n = {n_points:,} points" if n_points else ""
    print(f"  {stat:<14s} {len(sig_combined):5d} significant "
          f"({len(sig_pos_df)} positive, {len(sig_neg_df)} negative) "
          f"of {len(d):,} tested at FDR < {fdr_thresh} and "
          f"|rho| >= {rho_thresh}{npts}")

    mech = set(mechanomics or [])
    mech_sig = sig_combined[sig_combined["gene"].isin(mech)].copy()
    mech_pos = mech_sig[mech_sig["rho"] > 0].sort_values("rho", ascending=False)
    mech_neg = mech_sig[mech_sig["rho"] < 0].sort_values("rho", ascending=True)
    if mech:
        print(f"                 Mechanomics among them: {len(mech_sig)} "
              f"({len(mech_pos)} positive, {len(mech_neg)} negative)")

    label_df = pd.concat([mech_pos.head(MECHANOMICS_LABEL_N),
                          mech_neg.head(MECHANOMICS_LABEL_N)],
                         ignore_index=True)

    fig, ax = plt.subplots(figsize=(1.378, 1.575))
    fig.set_layout_engine("constrained")
    ax.scatter(d.loc[nonsig_mask, "rho"], d.loc[nonsig_mask, "neglog10_fdr"],
               c="lightgrey", s=VOLCANO_POINT_SIZE_NS, alpha=0.5,
               label="Not significant", rasterized=SVG_RASTERIZED_SCATTER)
    ax.scatter(d.loc[sig_pos_mask, "rho"], d.loc[sig_pos_mask, "neglog10_fdr"],
               c=STIFF_COL, s=VOLCANO_POINT_SIZE_SIG, alpha=0.8,
               label=f"FDR<{fdr_thresh}, rho>={rho_thresh:.2f} "
                     f"(n={int(sig_pos_mask.sum())})",
               rasterized=SVG_RASTERIZED_SCATTER)
    ax.scatter(d.loc[sig_neg_mask, "rho"], d.loc[sig_neg_mask, "neglog10_fdr"],
               c=SOFT_COL, s=VOLCANO_POINT_SIZE_SIG, alpha=0.8,
               label=f"FDR<{fdr_thresh}, rho<=-{rho_thresh:.2f} "
                     f"(n={int(sig_neg_mask.sum())})",
               rasterized=SVG_RASTERIZED_SCATTER)

    if len(mech_sig):
        ax.scatter(mech_sig["rho"], mech_sig["neglog10_fdr"],
                   facecolors="none", edgecolors="black",
                   s=MECHANOMICS_OUTLINE_SIZE,
                   linewidths=MECHANOMICS_OUTLINE_LW, zorder=5,
                   label=f"Mechanomics (n={len(mech_sig)})")

    ax.axhline(-np.log10(fdr_thresh), color="k", ls="--", lw=1.0, alpha=0.4)
    for v in (-rho_thresh, rho_thresh):
        ax.axvline(v, color="k", ls="--", lw=0.8, alpha=0.3)

    if SYMMETRIC_VOLCANO_X:
        xmax = max(float(d["rho"].abs().max()), rho_thresh)
        ax.set_xlim(-xmax * 1.08, xmax * 1.08)
    ax.set_ylim(bottom=0)

    texts = []
    for _, row in label_df.iterrows():
        texts.append(ax.text(row["rho"], row["neglog10_fdr"], row["gene"],
                             fontsize=VOLCANO_LABEL_FONTSIZE,
                             fontweight="bold", zorder=7))
    if texts:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.6, alpha=0.7),
            expand_points=(1.2, 1.4), expand_text=(1.2, 1.4),
            force_text=(0.5, 0.8), force_points=(0.3, 0.5))

    # Single short lines. Two-line labels cost ~4 mm of a 40 mm panel each,
    # and the target and statistic are both already in the filename.
    ax.set_xlabel("partial Spearman rho")
    ax.set_ylabel(r"$-\log_{10}$ FDR")
    ax.set_title(f"{group_label} \u00b7 {STAT_SHORT.get(stat, stat)}")
    ax.xaxis.set_major_locator(MaxNLocator(VOLCANO_MAX_TICKS))
    ax.yaxis.set_major_locator(MaxNLocator(VOLCANO_MAX_TICKS))
    if VOLCANO_LEGEND:
        leg = ax.legend(frameon=False, fontsize=LEGEND_FONT, loc="upper left")
        for lh in getattr(leg, "legend_handles",
                          getattr(leg, "legendHandles", [])):
            try:
                lh.set_alpha(1.0)
            except Exception:
                pass
    despine(ax)
    save_figure(fig, make_plot_stem(1, group_label, f"volcano_{stat}",
                                    target_col), out_dir)
    plt.close(fig)

    return d, sig_pos_df, sig_neg_df, sig_combined, sig_genes

# =============================================================================
# STEP 5: SIGNIFICANT GENE EXPORT AND MATRISOME
# =============================================================================

def export_significant_gene_reference_table(lognorm, gene_names, pmeta,
                                            sig_genes, out_dir, target_col,
                                            group_label="global"):
    """Point-level expression of the top significant genes, with metadata."""
    if not sig_genes:
        print(f" No significant genes for {group_label}; "
              f"skipping reference table.")
        return

    genes_out = list(sig_genes)[:REFERENCE_TABLE_MAX_GENES]
    if len(sig_genes) > REFERENCE_TABLE_MAX_GENES:
        print(f" Reference table capped at {REFERENCE_TABLE_MAX_GENES} of "
              f"{len(sig_genes)} significant genes.")

    idx = {g: i for i, g in enumerate(gene_names)}
    cols = [g for g in genes_out if g in idx]
    if not cols:
        return
    sub = np.column_stack([
        np.asarray(lognorm[:, idx[g]].todense()).ravel() for g in cols])
    expr_sub = pd.DataFrame(sub, columns=cols, index=pmeta.index)

    meta_cols = [c for c in [target_col, "age_group", "animal", "tissue", "roi",
                             "majority_cluster", "majority_compartment",
                             "n_bins_used", "depth", "log_depth",
                             "x_mean", "y_mean"] if c in pmeta.columns]
    wide = pmeta[meta_cols].join(expr_sub)
    wide.index.name = "point_row"
    wide.to_csv(
        out_dir / f"{group_label}_{target_col}_significant_genes_pointlevel_wide.csv",
        index=True)

    if EXPORT_LONG_TABLE:
        long = expr_sub.reset_index().melt(
            id_vars="index", var_name="gene", value_name="expression")
        long = long.rename(columns={"index": "point_row"})
        long = long.merge(
            pmeta[meta_cols].reset_index().rename(columns={"index": "point_row"}),
            on="point_row", how="left")
        long.to_csv(
            out_dir / f"{group_label}_{target_col}_significant_genes_pointlevel_long.csv",
            index=False)

def export_matrisome_hits(sig_combined_df, out_dir, target_col,
                          group_label="global", matrisome_df=None):
    if not RUN_MATRISOME or matrisome_df is None:
        return
    if sig_combined_df is None or sig_combined_df.empty:
        print(f" No significant genes for {group_label}; "
              f"skipping matrisome annotation.")
        return

    mat = matrisome_df.copy()
    gcol = MATRISOME_COLS["gene"]
    mat_genes = set(mat[gcol].astype(str))

    hits = sig_combined_df.copy()
    hits["gene"] = hits["gene"].astype(str)
    hits = hits[hits["gene"].isin(mat_genes)].copy()
    if hits.empty:
        print(f" No matrisome hits for {group_label}, {target_col}.")
        return

    hits = hits.merge(mat, left_on="gene", right_on=gcol,
                      how="left").sort_values(["direction", "rho"],
                                              ascending=[True, False])
    hits.to_csv(out_dir / f"{group_label}_{target_col}_matrisome_hits.csv",
                index=False)

    ccol = MATRISOME_COLS["category"] or MATRISOME_COLS["division"]
    if ccol is not None and ccol in hits.columns:
        (hits.groupby(["direction", ccol], dropna=False)
         .size().reset_index(name="n_genes")
         .to_csv(out_dir / f"{group_label}_{target_col}_matrisome_hit_summary.csv",
                 index=False))
    print(f" Saved matrisome hits: {group_label}_{target_col}_matrisome_hits.csv")

def export_significant_only_rho_sheet(res, out_dir, target_col, group_label="global"):
    """Simple per-group table: significant genes and global/young/aged R."""
    if "gene" not in res.columns:
        print(f" [SKIP] R summary for {group_label}: no gene column.")
        return

    sig = {}
    for nm in ("global", "young", "aged"):
        rho_col = f"rho_{nm}"
        fdr_col = f"fdr_{nm}"
        if rho_col in res.columns and fdr_col in res.columns:
            rho = pd.to_numeric(res[rho_col], errors="coerce")
            fdr = pd.to_numeric(res[fdr_col], errors="coerce")
            sig[nm] = (fdr < FDR_THRESH) & (rho.abs() >= RHO_THRESH)
        else:
            sig[nm] = pd.Series(False, index=res.index)

    sig_any = sig["global"] | sig["young"] | sig["aged"]

    out = pd.DataFrame({
        "gene": res["gene"].astype(str),
        "R_global": pd.to_numeric(res.get("rho_global", np.nan),
                                  errors="coerce"),
        "R_young": pd.to_numeric(res.get("rho_young", np.nan),
                                 errors="coerce"),
        "R_aged": pd.to_numeric(res.get("rho_aged", np.nan),
                                errors="coerce"),
    })

    # Keep only significant genes, and blank non-significant strata.
    out.loc[~sig["global"], "R_global"] = np.nan
    out.loc[~sig["young"], "R_young"] = np.nan
    out.loc[~sig["aged"], "R_aged"] = np.nan
    out = out.loc[sig_any].reset_index(drop=True)

    xlsx_path = out_dir / f"{group_label}_{target_col}_significant_only_r_values.xlsx"
    try:
        out.to_excel(xlsx_path, index=False, sheet_name="R_values")
        print(f" Saved significant-only R table: {xlsx_path.name} "
              f"({len(out):,} gene(s))")
    except Exception as e:
        csv_path = out_dir / f"{group_label}_{target_col}_significant_only_r_values.csv"
        out.to_csv(csv_path, index=False)
        print(f" [WARN] Could not write Excel ({e}); wrote CSV "
              f"instead: {csv_path.name}")

def matrisome_fisher_test(res, stat, matrisome_df, mechanomics, out_dir,
                          target_col, group_label="global"):
    """
    Fisher exact enrichment with the CORRECT background: the genes actually
    tested in this group, not the whole annotated genome.

    Plain reading of the output. Suppose 10,000 genes were tested and 1,100 of
    them are matrisome (11%), and 1,204 genes came out significantly positive.
    By chance you would expect about 132 of the significant genes to be
    matrisome. If you observe 250, that is roughly a 1.9-fold enrichment and
    the p-value asks how surprising that is. This is what Enrichr's ORA was
    trying to do, except Enrichr uses the whole genome as the background and so
    inflates everything.
    """
    rho_col, fdr_col = f"rho_{stat}", f"fdr_{stat}"
    if rho_col not in res.columns:
        return None
    d = res.dropna(subset=[rho_col, fdr_col])
    universe = set(d["gene"].astype(str))
    if not universe:
        return None

    sig_up = set(d.loc[(d[fdr_col] < FDR_THRESH) & (d[rho_col] >= RHO_THRESH),
                       "gene"].astype(str))
    sig_dn = set(d.loc[(d[fdr_col] < FDR_THRESH) & (d[rho_col] <= -RHO_THRESH),
                       "gene"].astype(str))

    groups = []
    if matrisome_df is not None:
        mat = matrisome_df
        gcol = MATRISOME_COLS["gene"]
        groups.append(("all matrisome", set(mat[gcol].astype(str))))
        for col in (MATRISOME_COLS["division"], MATRISOME_COLS["category"]):
            if col is not None and col in mat.columns:
                for lv, sub in mat.groupby(col):
                    groups.append((str(lv), set(sub[gcol].astype(str))))
    if mechanomics:
        groups.append(("Mechanomics", set(mechanomics)))

    rows = []
    for name, gset in groups:
        in_uni = gset & universe
        if len(in_uni) < 5:
            continue
        for lbl, sig in (("up_with_stiffness", sig_up),
                         ("down_with_stiffness", sig_dn)):
            if len(sig) < 5:
                continue
            a = len(sig & in_uni)
            b = len(sig) - a
            c = len(in_uni) - a
            dd = len(universe) - len(sig) - c
            orr, p = fisher_exact([[a, b], [c, dd]], alternative="greater")
            rows.append(dict(group=name, direction=lbl, n_sig=len(sig),
                             n_group_tested=len(in_uni), n_overlap=a,
                             expected=len(sig) * len(in_uni) / len(universe),
                             fold_enrichment=(a / (len(sig) * len(in_uni)
                                                   / len(universe)))
                             if a else 0.0,
                             odds_ratio=orr, pval=p,
                             n_universe=len(universe)))
    if not rows:
        print(" Matrisome Fisher test: too few significant genes to test.")
        return None

    t = pd.DataFrame(rows)
    t["fdr"] = multipletests(t["pval"], method="fdr_bh")[1]
    t = t.sort_values("pval")
    t.to_csv(out_dir / f"{group_label}_{target_col}_{stat_disp(stat)}"
                       f"_matrisome_enrichment.csv",
             index=False)

    hit = t[t["fdr"] < FDR_THRESH]
    if len(hit):
        print(f" Matrisome / Mechanomics enrichment at FDR < {FDR_THRESH}:")
        print(hit[["group", "direction", "n_overlap", "expected",
                   "fold_enrichment", "odds_ratio", "fdr"]].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
    else:
        print(" No matrisome / Mechanomics group enriched at FDR < 0.05.")
    return t

# =============================================================================
# STEP 6: PER-GENE CORRELATION PANELS
# =============================================================================

def _std_resid(v, model):
    """
    Rank residuals of x and of the target, each scaled to unit standard
    deviation.

    Standardising is what makes the reference line honest. On unstandardised
    rank residuals the least-squares slope is rho x sd(ex)/sd(ey), so a cloud
    with rho = +0.30 can look near-vertical or near-flat depending on the
    residual spreads. Once both axes are in units of their own SD, the
    least-squares slope IS rho, and a line drawn at slope 0.30 is exactly the
    quoted number rather than an impression of it.
    """
    ex = partial_residuals(v, model)
    ey = model["ey"]
    sx, sy = np.std(ex), np.std(ey)
    if not np.isfinite(sx) or sx <= 0 or not np.isfinite(sy) or sy <= 0:
        return None, None
    return ex / sx, ey / sy

# =============================================================================
# ADJUSTED CORRELATION PANEL
#
# The readable version of the added-variable plot. Same axes as the raw panel,
# stiffness in Pa on x and normalised expression on y, but with the sequencing
# depth artefact removed from the expression values instead of from the axes.
#
# In a simulation where depth and stiffness were correlated, a gene with a
# small real effect read rho = +0.75 raw and rho = +0.09 once depth was taken
# out. Almost all of the raw slope was deeper spots reading higher for
# everything, which is exactly what this panel is for.
# =============================================================================

PLOT_ADJUSTED_PANEL = True     # write the extra figure per gene
ADJ_PANEL_W_IN = 1.575         # one panel of the existing 3.15 in pair
ADJ_PANEL_H_IN = 1.378
ADJ_PANEL_Y_FLOOR_ZERO = True  # start y at 0 where the data allow

# The black line.
#   "rho"  slope implied by the quoted rho, i.e. rho x sd(y)/sd(x), drawn
#          through the centroid. This is the line that REPRESENTS the number
#          printed on the panel, which is what you asked for.
#   "ols"  ordinary least squares fit to the adjusted points. Very close to
#          the above, but it is its own fit rather than the quoted rho.
ADJ_PANEL_LINE = "rho"

# Line weights and the in-plot label styling.
ADJ_LINE_LW = 0.6             # the coloured stratum lines. Matched to the
                              # black line below, per your request that all
                              # trend lines read at the same weight.
ADJ_LINE_LW_BLACK = 0.6       # the selected statistic, kept heavier so it
                              # still reads as the primary line
ADJ_LINE_DARKEN = 0.18        # how much darker the teal/purple lines are than
                              # their own points, as a FRACTION of the colour's
                              # own lightness. Fractional, not a flat
                              # subtraction: the teal #007A7A already sits at
                              # lightness 0.24, so subtracting a flat 0.18 left
                              # #001e1e, which renders as black. Scaling gives
                              # #006464, still visibly teal.
ADJ_LABEL_FONT = 5.5          # the in-plot rho labels
# Shown in place of a rho when that stratum did NOT pass the gate. "NS", not
# "NC": the correlation was measured, it just did not clear FDR < FDR_THRESH
# and |rho| >= RHO_THRESH. Caption it as "NS: no significant correlation in
# that age group".
ADJ_NS_LABEL = "NS"

# On a GLOBAL panel, a stratum that passes the gate gets darker dots and a
# trend line; one that does not gets lighter dots and no line. youngcorr and
# agedcorr panels are untouched and keep the base colours.
ADJ_SIG_AWARE_GLOBAL = True
ADJ_DOT_DARKEN_SIG = 0.22     # significant stratum, darker than the base
ADJ_DOT_LIGHTEN_NS = 0.28     # non-significant stratum, lighter than the base

ADJ_LABEL_SHORT = {"young": "Y", "aged": "A", "global": "both",
                   "within_animal": "within"}
ADJ_YLABEL = "Log-normalised expression"   # verified against the matrix:
                              # counts are CPM-like scaled then np.log1p,
                              # so these values really are log-normalised,
                              # with the depth component subtracted.
ADJ_XLABEL = None             # None = the short auto label

# matplotlib's `s` is AREA in points squared, not diameter. POINT_SIZE is 4,
# so 2.0 halves the area and makes each dot about 71% of its current width.
# For dots half as WIDE, set this to 1.0 instead. Only the corrected panel
# uses it; every other panel still uses POINT_SIZE.
ADJ_POINT_SIZE = MARKER_SIZE_SMALL   # same "small dot" as the volcanoes

# The LEFT-hand raw panel of the two-panel figure.
#   PLOT_RAW_PANEL = False writes ONLY the corrected panel, so the raw
#   plus rank-residual figure stops being produced at all.
#
# Now False. It was producing a second figure per gene for EVERY statistic,
# whose left half is an unadjusted OLS fit that does not match the quoted rho
# and whose right half is in residual SD units. Set back to True only if you
# want the added-variable diagnostic again.
PLOT_RAW_PANEL = False
RAW_LABEL_FONT = 6

# "Normalised expression" is 0.93 in long at 6 pt, but the axes in a 1.378 in
# panel are only about half an inch tall, so the label overran the figure and
# was cut off at both ends. Rather than shorten your wording, the y label is
# shrunk until it fits the measured axes height. Set to a number to force a
# fixed size instead, in which case it may clip again.
ADJ_YLABEL_FONT = "auto"
ADJ_YLABEL_FONT_MIN = 4.0     # never go below this; report instead

# -----------------------------------------------------------------------------
# AXES: fixed x range, "closed" axes, sensible y ticks
#
# Both axes now END on a labelled tick, so the spine stops at a number instead
# of running past the last one.
#
# x is fixed across every gene panel so the panels are directly comparable.
# ADJ_XLIM_TARGETS restricts that to the targets where a fixed range is
# meaningful; Loss and Viscosity are on completely different scales and would
# be squashed into a corner by a 0-5 range.
# If any point falls outside ADJ_XLIM the limit is WIDENED and a warning is
# printed once. Silently clipping a real observation off the panel would be
# worse than an inconsistent axis.
# -----------------------------------------------------------------------------
ADJ_XLIM = (0.0, 5.0)                    # log10 E (Pa); None = auto
ADJ_X_TICKS = [2.5, 5.0]                 # None = let matplotlib choose
ADJ_XLIM_TARGETS = ("Elasticity_l_Pa", "Elasticity_eff_Pa")

# y is per gene, because expression range is per gene. Only the RULE is shared:
# round the top up to a 0.5 (or 1.0 for larger ranges) increment, then split the
# span with the largest "nice" step that gives 2 to ADJ_Y_MAX_INTERVALS gaps.
#   span 6.0  -> step 3.0   -> 0, 3, 6
#   span 1.5  -> step 0.5   -> 0, 0.5, 1.0, 1.5
#   span 2.0  -> step 1.0   -> 0, 1, 2
ADJ_Y_NICE_AXIS = True
ADJ_Y_MAX_INTERVALS = 3       # so at most 4 tick labels on a 1.4 in panel
# Allowed tick steps, as mantissas scaled by powers of ten. 1, 1.5, 2, 2.5, 3
# and 5 are the increments that read as "round" on an axis; 0.5 and 1.0 fall
# out of this automatically (0.5 = 5 x 10^-1, 1.0 = 1 x 10^0).
ADJ_Y_STEP_MANTISSAS = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]

# Every y tick label printed to this many decimals, so "3" and "1.5" do not
# have different widths. On its own this is NOT enough to make the plotted
# area identical across panels (see ADJ_FIXED_AXES_RECT below): "0.5" and
# "15.0" still differ by a digit. It removes the most common source of the
# shift, and the fixed rectangle removes the rest.
ADJ_Y_TICK_DECIMALS = 1

# -----------------------------------------------------------------------------
# IDENTICAL PLOTTED AREA ACROSS PANELS
#
# Constrained layout sizes the axes around whatever the tick labels happen to
# need, so a gene whose axis reads 0.0-1.5 got a WIDER plotted area than one
# reading 0.0-15.0, and the two panels no longer scale against each other.
# With ADJ_FIXED_AXES_RECT the axes rectangle is set explicitly in figure
# fractions and the layout engine is switched off, so every panel has a
# plotted area of exactly the same size and position.
#
# ADJ_AXES_RECT is (left, bottom, width, height). The margins have to hold:
#   left    y tick labels + tick marks + the rotated y label
#   bottom  x tick labels + tick marks + the x label
#   top     the gene title
# If a label ever looks cramped, widen the margin here rather than turning the
# fixed rectangle off; that is the only setting that guarantees the panels
# match. Set ADJ_FIXED_AXES_RECT = False to go back to constrained layout.
# -----------------------------------------------------------------------------
ADJ_FIXED_AXES_RECT = True
ADJ_AXES_RECT = (0.300, 0.270, 0.640, 0.600)

# -----------------------------------------------------------------------------
# GLOBAL PANELS: one population, one line
#
# A globalcorr panel shows the POOLED correlation. Colouring the dots by age
# and drawing per-stratum lines on it invited the panel to be read as three
# results at once, and the age-specific numbers already have their own panels.
# So: every point black at one alpha, one black line, one number.
# Set ADJ_GLOBAL_ALL_BLACK = False to go back to the two-colour version with
# significance-aware shading.
# -----------------------------------------------------------------------------
ADJ_GLOBAL_ALL_BLACK = True
# Grey, not black. At 900 overlapping points a black cloud saturates and the
# black trend line disappears into it. Mid grey keeps the cloud readable and
# leaves the line as the darkest thing on the panel.
ADJ_GLOBAL_DOT_COL = "#7F7F7F"
ADJ_GLOBAL_DOT_ALPHA_FILL = 0.30   # one alpha for every point, both ages
ADJ_GLOBAL_DOT_ALPHA_EDGE = 0.55

# The trend line for the statistic the panel is built on. Global stays black;
# the age-stratified panels take a darker version of their own stratum colour
# so youngcorr and agedcorr are distinguishable at a glance.
ADJ_STAT_LINE_DARKEN = 0.32
ADJ_LABEL_MATCHES_LINE = True   # the r label takes the trend-line colour

# The printed correlation. "both +0.10" implied a comparison the panel is not
# making, so it now reads "r = +0.10". Applies to every statistic.
ADJ_R_LABEL_FMT = "r = {r:+.2f}"

def _lighten(hex_colour, amount):
    """
    Blend toward white in RGB, not "raise lightness" in HLS.

    Raising HLS lightness holds saturation, so the dark saturated teal
    #007A7A came out as #00E7E7, a near-neon cyan, rather than a pale teal.
    An RGB blend toward white desaturates as it lightens, which is what "a
    slightly lighter teal" actually means: #007A7A -> #47B2B2.
    """
    r, g, b = mcolors.to_rgb(hex_colour)
    f = float(np.clip(amount, 0.0, 1.0))
    return mcolors.to_hex((r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f))


def _stratum_is_sig(row, nm):
    """Did this stratum clear the SAME gate the volcano and the CSV use?"""
    if row is None:
        return False
    r = row.get(f"rho_{nm}", np.nan)
    f = row.get(f"fdr_{nm}", np.nan)
    return bool(np.isfinite(r) and np.isfinite(f)
                and f < FDR_THRESH and abs(r) >= RHO_THRESH)


def _darken(hex_colour, amount=None):
    """Same hue, lower lightness, so a line reads against its own points."""
    amount = ADJ_LINE_DARKEN if amount is None else amount
    h_, l_, s_ = colorsys.rgb_to_hls(*mcolors.to_rgb(hex_colour))
    return mcolors.to_hex(colorsys.hls_to_rgb(h_, l_ * (1.0 - amount), s_))

_YLABEL_WARNED = set()
_XLIM_WARNED = set()


def nice_y_axis(vmin, vmax, floor_zero=True):
    """
    A y range that ENDS on a labelled tick, split into sensible intervals.

    Returns (lo, hi, ticks), or (None, None, None) to leave matplotlib's own
    locator alone rather than invent a range that does not contain the data.

    Method. Candidate steps are k x 10^e for k in ADJ_Y_STEP_MANTISSAS, over a
    range of exponents around the data span. For each step the axis is closed
    at the smallest multiple that covers the data, and the candidate is kept
    when the number of intervals is between 2 and ADJ_Y_MAX_INTERVALS. Of the
    survivors the TIGHTEST axis wins, and ties go to the one with the fewest
    labels, because a 1.4 in panel cannot carry many.

    Verified behaviour (see the self-test printed by --check-axes):
      data max 5.8 -> 0, 3, 6                (top rounded to a 1.0 increment)
      data max 1.4 -> 0, 0.5, 1.0, 1.5       (top rounded to a 0.5 increment)
      data max 1.8 -> 0, 1, 2
      data max 2.4 -> 0, 1.5, 3.0
    Zero is always a tick when the data are non-negative, so panels for
    different genes still share a common origin.
    """
    vmin = float(vmin)
    vmax = float(vmax)
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
        return None, None, None

    base_lo = 0.0 if (floor_zero and vmin >= 0) else min(0.0, vmin)
    span0 = max(vmax - base_lo, abs(vmin), 1e-12)
    e0 = int(np.floor(np.log10(span0)))

    best = None
    for e in range(e0 - 2, e0 + 3):
        for k in ADJ_Y_STEP_MANTISSAS:
            step = k * (10.0 ** e)
            if step <= 0:
                continue
            n_hi = int(np.ceil(vmax / step - 1e-9))
            n_lo = 0 if (floor_zero and vmin >= 0) else \
                int(np.ceil(max(0.0, -vmin) / step - 1e-9))
            n_hi = max(n_hi, 0)
            n_tot = n_hi + n_lo
            if not (2 <= n_tot <= ADJ_Y_MAX_INTERVALS):
                continue
            lo = -n_lo * step
            hi = n_hi * step
            if hi <= lo:
                continue
            cand = (round(hi - lo, 12), n_tot, step, lo, hi)
            if best is None or cand[:2] < best[:2]:
                best = cand
    if best is None:
        return None, None, None

    _, n_tot, step, lo, hi = best
    ticks = [lo + i * step for i in range(n_tot + 1)]
    # Kill floating-point dust such as 0.30000000000000004 in the tick labels.
    ticks = [float(np.round(t, 10)) for t in ticks]
    return float(np.round(lo, 10)), float(np.round(hi, 10)), ticks


def _fit_ylabel_font(fig, ax, text):
    """
    Largest font at which a rotated y label still fits the axes height.

    Measures the drawn axes rather than guessing, then binary-steps down from
    AXIS_FONT. Text width scales linearly with point size, so one measurement
    at the current size is enough to solve for the fitting size directly.
    """
    if ADJ_YLABEL_FONT != "auto":
        return ADJ_YLABEL_FONT
    fig.canvas.draw()
    avail = ax.get_window_extent().height / fig.dpi * 0.98
    # FontProperties() with no family argument inherits the rcParams font.
    # Passing the family as a bare string trips matplotlib's fontconfig parser
    # on generic names such as "sans-serif".
    tp = TextPath((0, 0), str(text), size=AXIS_FONT, prop=FontProperties())
    need = tp.get_extents().width / 72.0
    if need <= avail or need <= 0:
        return AXIS_FONT
    fs = max(ADJ_YLABEL_FONT_MIN, AXIS_FONT * avail / need)
    if fs <= ADJ_YLABEL_FONT_MIN and text not in _YLABEL_WARNED:
        _YLABEL_WARNED.add(text)
        print(f"  NOTE: y label '{text}' does not fit a "
              f"{ADJ_PANEL_H_IN} in panel even at {ADJ_YLABEL_FONT_MIN} pt "
              f"and will be clipped. Shorten ADJ_YLABEL or raise "
              f"ADJ_PANEL_H_IN.")
    return fs

def _italic(s):
    """Gene symbol in italics, plain if it holds characters mathtext dislikes."""
    t = str(s)
    return f"$\\it{{{t}}}$" if t.replace("_", "").isalnum() else t

def depth_adjusted_expression(v, log_depth):
    """
    Expression corrected to a common sequencing depth.

    Fits expression on log10 depth and subtracts ONLY the depth component,
    leaving the overall mean where it was. Units and scale are unchanged, so
    the y axis is still normalised expression and still starts near zero.
    """
    v = np.asarray(v, float)
    d = np.asarray(log_depth, float)
    ok = np.isfinite(v) & np.isfinite(d)
    if int(ok.sum()) < 3 or np.ptp(d[ok]) == 0:
        return v.copy()
    X = np.column_stack([np.ones(int(ok.sum())), d[ok]])
    beta = np.linalg.lstsq(X, v[ok], rcond=None)[0]
    out = v.copy()
    out[ok] = v[ok] - beta[1] * (d[ok] - d[ok].mean())
    return out

def _rho_line(xv, yv, rho):
    """
    The line a correlation of `rho` implies, in the data's own units.

    A correlation is unit-free; to draw it on real axes it has to be scaled by
    the spread of each axis. slope = rho x sd(y)/sd(x), through the centroid.
    So the drawn steepness IS the printed number, expressed in Pa and
    expression units rather than in standard deviations.
    """
    ok = np.isfinite(xv) & np.isfinite(yv)
    if int(ok.sum()) < 3 or not np.isfinite(rho):
        return None, None
    sx, sy = np.std(xv[ok]), np.std(yv[ok])
    if sx <= 0:
        return None, None
    slope = rho * sy / sx
    xs = np.linspace(float(np.nanmin(xv[ok])), float(np.nanmax(xv[ok])), 50)
    ys = yv[ok].mean() + slope * (xs - xv[ok].mean())
    return xs, ys

def plot_adjusted_correlation(y, v_adj, ages, gene, target_col, out_fig,
                              group_label="global", stat="global",
                              stem_tag="", row=None):
    """
    Depth-adjusted expression against stiffness, on readable axes.

    Same formatting as the raw panel so the two sit side by side. The drawn
    lines are the QUOTED correlations expressed in these units, not separate
    fits: slope = rho x sd(y)/sd(x) through the centroid.
    """
    if ADJ_FIXED_AXES_RECT:
        # Explicit rectangle, layout engine OFF. Constrained layout would
        # immediately resize the axes around the tick labels again, which is
        # the whole problem this avoids.
        fig = plt.figure(figsize=(ADJ_PANEL_W_IN, ADJ_PANEL_H_IN))
        fig.set_layout_engine("none")
        ax = fig.add_axes(ADJ_AXES_RECT)
    else:
        fig, ax = plt.subplots(figsize=(ADJ_PANEL_W_IN, ADJ_PANEL_H_IN))
        fig.set_layout_engine("constrained")

    keep = (["young"] if stat == "young"
            else ["aged"] if stat == "aged" else ["young", "aged"])

    # Trend-line colour for the statistic this panel is built on. Global stays
    # black because it is one pooled population; the stratified panels take a
    # darker version of their own colour so youngcorr and agedcorr are
    # distinguishable without reading the title.
    stat_line_col = {"young": _darken(YOUNG_COL, ADJ_STAT_LINE_DARKEN),
                     "aged": _darken(AGED_COL, ADJ_STAT_LINE_DARKEN)
                     }.get(stat, "black")

    global_flat = (stat == "global" and ADJ_GLOBAL_ALL_BLACK)

    # On a global panel the dot shade used to encode whether each stratum
    # passed its own gate. With ADJ_GLOBAL_ALL_BLACK the panel shows the pooled
    # result only, so every point is the same colour at the same alpha.
    sig_by = {nm: _stratum_is_sig(row, nm) for nm in ("young", "aged")}
    shade = {}
    for age, col in [("young", YOUNG_COL), ("aged", AGED_COL)]:
        if global_flat:
            shade[age] = ADJ_GLOBAL_DOT_COL
        elif stat == "global" and ADJ_SIG_AWARE_GLOBAL:
            shade[age] = (_darken(col, ADJ_DOT_DARKEN_SIG) if sig_by[age]
                          else _lighten(col, ADJ_DOT_LIGHTEN_NS))
        else:
            shade[age] = col

    a_fill = ADJ_GLOBAL_DOT_ALPHA_FILL if global_flat else POINT_ALPHA_FILL
    a_edge = ADJ_GLOBAL_DOT_ALPHA_EDGE if global_flat else POINT_ALPHA_EDGE

    for age in ("young", "aged"):
        if age not in keep:
            continue
        m = ages == age
        if m.sum() < 2:
            continue
        col = shade[age]
        ax.scatter(y[m], v_adj[m],
                   facecolors=to_rgba(col, a_fill),
                   edgecolors=to_rgba(col, a_edge),
                   s=ADJ_POINT_SIZE, linewidths=POINT_LINEWIDTH,
                   label=age.capitalize(),
                   rasterized=SVG_RASTERIZED_CORRELATION)

    sel = np.isin(ages, keep)
    r_sel = row.get(f"rho_{stat}", np.nan) if row is not None else np.nan

    # Per-stratum lines belong on a panel that is ABOUT the strata. A flat
    # global panel draws the pooled line only.
    if (stat == "global" and not global_flat
            and RHO_LINES_PER_STRATUM and row is not None):
        for nm in ("young", "aged"):
            # A line is drawn ONLY where that stratum cleared the gate. Drawing
            # one for a non-significant stratum would assert a correlation the
            # analysis declined to call.
            if ADJ_SIG_AWARE_GLOBAL and not sig_by[nm]:
                continue
            m = ages == nm
            rr = row.get(f"rho_{nm}", np.nan)
            if m.sum() >= 3 and np.isfinite(rr):
                xs, ys = _rho_line(y[m], v_adj[m], rr)
                if xs is not None:
                    ax.plot(xs, ys, color=_darken(shade[nm]), lw=ADJ_LINE_LW,
                            alpha=1.0, zorder=5)
    if sel.sum() >= 3:
        if ADJ_PANEL_LINE == "rho" and np.isfinite(r_sel):
            xs, ys = _rho_line(y[sel], v_adj[sel], r_sel)
        else:
            z = np.polyfit(y[sel], v_adj[sel], 1)
            xs = np.linspace(y[sel].min(), y[sel].max(), 50)
            ys = np.polyval(z, xs)
        if xs is not None:
            ax.plot(xs, ys, color=stat_line_col, lw=ADJ_LINE_LW_BLACK,
                    alpha=1.0, zorder=6)

    labels = []
    if row is not None:
        if stat == "global" and not global_flat:
            for nm in ("young", "aged"):
                r = row.get(f"rho_{nm}", np.nan)
                short = ADJ_LABEL_SHORT.get(nm, nm)
                if ADJ_SIG_AWARE_GLOBAL and not sig_by[nm]:
                    # The rho is deliberately NOT printed: quoting a number the
                    # analysis refused to call significant invites it to be
                    # read as a result.
                    labels.append((f"{short} {ADJ_NS_LABEL}",
                                   _darken(shade[nm])))
                elif np.isfinite(r):
                    labels.append((f"{short} {r:+.2f}", _darken(shade[nm])))
        if np.isfinite(r_sel):
            labels.append((ADJ_R_LABEL_FMT.format(r=r_sel),
                           stat_line_col if ADJ_LABEL_MATCHES_LINE else "black"))

    # ---- axes -------------------------------------------------------------
    # x first, because the label band below is measured against the drawn axes.
    if ADJ_XLIM is not None and (ADJ_XLIM_TARGETS is None
                                 or target_col in ADJ_XLIM_TARGETS):
        x_lo, x_hi = float(ADJ_XLIM[0]), float(ADJ_XLIM[1])
        d_lo = float(np.nanmin(y[sel])) if sel.sum() else x_lo
        d_hi = float(np.nanmax(y[sel])) if sel.sum() else x_hi
        if d_lo < x_lo or d_hi > x_hi:
            # Never clip a real observation off the panel to keep a tidy axis.
            key = (target_col, round(d_lo, 3), round(d_hi, 3))
            if key not in _XLIM_WARNED:
                _XLIM_WARNED.add(key)
                print(f" [NOTE] {target_col}: data span {d_lo:.2f} to "
                      f"{d_hi:.2f} falls outside ADJ_XLIM {ADJ_XLIM}. The "
                      f"limit was widened for the affected panels, so those "
                      f"are NOT on the shared x range.")
            x_lo, x_hi = min(x_lo, d_lo), max(x_hi, d_hi)
            ax.set_xlim(x_lo, x_hi)
            ax.xaxis.set_major_locator(MaxNLocator(VOLCANO_MAX_TICKS))
        else:
            ax.set_xlim(x_lo, x_hi)
            if ADJ_X_TICKS:
                ax.set_xticks([t for t in ADJ_X_TICKS if x_lo <= t <= x_hi])
            else:
                ax.xaxis.set_major_locator(MaxNLocator(VOLCANO_MAX_TICKS))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(VOLCANO_MAX_TICKS))

    # y: reserve the label band FIRST, then close the axis on a round number,
    # so the top of the spine is a labelled tick rather than wherever the band
    # happened to end.
    v_sel = v_adj[sel] if sel.sum() else v_adj
    y_lo_data = float(np.nanmin(v_sel)) if len(v_sel) else 0.0
    y_hi_data = float(np.nanmax(v_sel)) if len(v_sel) else 1.0
    if ADJ_PANEL_Y_FLOOR_ZERO:
        y_lo_data = min(0.0, y_lo_data)

    band = 0.0
    if labels:
        fig.canvas.draw()
        ax_h_in = ax.get_window_extent().height / fig.dpi
        step = (ADJ_LABEL_FONT * 1.35 / 72.0) / max(ax_h_in, 1e-6)
        band = (y_hi_data - y_lo_data) * (step * len(labels) + 0.04)

    lo_n, hi_n, ticks = (nice_y_axis(y_lo_data, y_hi_data + band,
                                     floor_zero=ADJ_PANEL_Y_FLOOR_ZERO)
                         if ADJ_Y_NICE_AXIS else (None, None, None))
    if ticks is not None:
        ax.set_ylim(lo_n, hi_n)
        ax.set_yticks(ticks)
    else:
        ax.set_ylim(y_lo_data, y_hi_data + band)
        ax.yaxis.set_major_locator(MaxNLocator(VOLCANO_MAX_TICKS))
    if ADJ_Y_TICK_DECIMALS is not None:
        # Fixed decimals, so "3" and "1.5" cannot have different label widths.
        ax.yaxis.set_major_formatter(
            FormatStrFormatter(f"%.{int(ADJ_Y_TICK_DECIMALS)}f"))

    if labels:
        # Labels go on the side where the trend line is LOW, so the line cannot
        # run up into them.
        fig.canvas.draw()
        ax_h_in = ax.get_window_extent().height / fig.dpi
        step = (ADJ_LABEL_FONT * 1.35 / 72.0) / max(ax_h_in, 1e-6)
        left = (r_sel >= 0) if np.isfinite(r_sel) else True
        fx, ha = (0.03, "left") if left else (0.97, "right")
        for i, (txt, col) in enumerate(labels):
            ax.text(fx, 0.985 - step * i, txt, transform=ax.transAxes,
                    fontsize=ADJ_LABEL_FONT, color=col, ha=ha, va="top")

    ax.set_xlabel(ADJ_XLABEL or "log$_{10}$ $E$ (Pa)")
    ax.set_ylabel(ADJ_YLABEL, fontsize=_fit_ylabel_font(fig, ax, ADJ_YLABEL))
    if PANEL_SUBTITLES:
        ax.set_title("depth-adjusted", fontsize=SUBTITLE_FONT)
    if PANEL_LEGEND:
        ax.legend(frameon=False, fontsize=LEGEND_FONT)
    # NOTE: the MaxNLocator calls that used to sit here have been removed.
    # They ran AFTER the tick lists above and silently threw them away, which
    # is why an earlier attempt at fixed x ticks appeared to do nothing.
    despine(ax)

    fig.suptitle(f"{_italic(gene)}, correlation ({stat_disp(stat)})",
                 fontsize=TITLE_FONT)
    save_figure(fig, make_plot_stem(3, f"{group_label}{stem_tag}",
                                    "correlation_adj", target_col, gene=gene),
                out_fig)
    plt.close(fig)

def plot_gene_correlations(lognorm, raw_sum, gene_names, pmeta, res, out_dir,
                           target_col, genes=None, group_label="global",
                           stat="global", stem_tag=""):
    """
    Two panels per gene, using the SAME per-gene point subset as the
    correlation statistics.

    LEFT   raw point-level expression against the target, split by age. NOTHING
           is adjusted here, so the coloured lines are ordinary least squares
           on the raw values and do NOT correspond to the quoted rho, which is
           a depth-adjusted partial Spearman. The panel is labelled to say so.
    RIGHT  added-variable plot on STANDARDISED rank residuals, so the drawn
           slope equals the quoted rho exactly. The reference lines are the
           actual correlation values: black for the statistic this panel was
           selected on, teal and purple for young and aged when those models
           exist.

    stat     which statistic selected these genes: "global", "young", "aged".
             It decides which model the right panel is built from, so a gene
             picked on rho_young is displayed against the young model rather
             than the pooled one.
    stem_tag appended to the filename, so the three statistics do not overwrite
             each other.
    """
    print("\n" + "=" * 60)
    print(f"Correlation panels: {group_label}, {target_col}")
    print("=" * 60)

    out_fig = out_dir / "gene_correlations"
    idx = {g: i for i, g in enumerate(gene_names)}

    # No fallback to MANUAL_GENE_PLOT_LIST. An empty list means there was
    # nothing worth drawing, and `genes or MANUAL_GENE_PLOT_LIST` would have
    # quietly substituted the whole curated list instead.
    if not genes:
        print(" No genes supplied; no correlation panels drawn.")
        return
    genes = list(genes)
    available = [g for g in genes if g in idx]
    missing = [g for g in genes if g not in idx]
    if missing:
        print(f" Not in the tested gene set ({len(missing)}): "
              f"{missing[:12]}{' ...' if len(missing) > 12 else ''}")
    if not available:
        print(" No genes available to plot.")
        return
    if len(available) > MAX_GENE_PANELS:
        print(f" Capped at {MAX_GENE_PANELS} panels (of {len(available)} "
              f"requested).")
        available = available[:MAX_GENE_PANELS]

    y = pmeta[target_col].values.astype(float)
    if LOG_TARGET and target_col in LOG_TARGETS:
        y = np.log10(np.maximum(y, 1e-12))
    ages = pmeta["age_group"].values
    log_depth_all = (pmeta["log_depth"].values
                     if "log_depth" in pmeta.columns else None)
    if PLOT_ADJUSTED_PANEL and log_depth_all is None:
        print(" [NOTE] pmeta has no 'log_depth'; the depth-adjusted panel is "
              "skipped.")

    # The right panel is built from the model that SELECTED these genes, so a
    # gene chosen on rho_young is shown against the young model, not the
    # pooled one. Falls back to global if that stratum model does not exist.
    stats_by_gene = res.set_index("gene") if res is not None else None
    Rc = raw_sum.tocsc()
    animals_all = (pmeta["animal"].values if "animal" in pmeta.columns
                   else np.array(["one"] * len(pmeta)))
    is_young_all = (ages == "young")

    plotted = 0
    skipped = 0

    for gene in available:
        v_all = np.asarray(lognorm[:, idx[gene]].todense()).ravel()
        raw_v = np.asarray(Rc[:, idx[gene]].todense()).ravel()
        expr_mask = expression_subset_mask(raw_v)
        n_expr = int(expr_mask.sum())

        if n_expr < MIN_POINTS_PER_GENE_CORR:
            skipped += 1
            continue

        y_g = y[expr_mask]
        v_g = v_all[expr_mask]
        ages_g = ages[expr_mask]
        log_depth_g = (log_depth_all[expr_mask] if log_depth_all is not None
                       else None)
        animals_g = animals_all[expr_mask]
        is_young_g = is_young_all[expr_mask]
        row_g = (stats_by_gene.loc[gene]
                 if stats_by_gene is not None and gene in stats_by_gene.index
                 else None)

        if not PLOT_RAW_PANEL:
            # Corrected panel only. The two-panel figure is never built rather
            # than built and thrown away.
            if PLOT_ADJUSTED_PANEL and log_depth_g is not None:
                plot_adjusted_correlation(
                    y_g, depth_adjusted_expression(v_g, log_depth_g),
                    ages_g, gene, target_col, out_fig,
                    group_label=group_label, stat=stat, stem_tag=stem_tag,
                    row=row_g)
                plotted += 1
            continue

        if log_depth_g is None:
            print(f" [NOTE] skipping residual panel for {gene}: no 'log_depth'.")
            continue

        models_gene = build_models(y_g, log_depth_g, animals_g, is_young_g,
                                   n_expr)
        model_key = stat if stat in models_gene else "global"
        model = models_gene[model_key]
        ey = model["ey"]

        # Which rows in the per-gene subset this model was fitted on.
        if model_key == "young":
            rows = (ages_g == "young")
        elif model_key == "aged":
            rows = (ages_g == "aged")
        else:
            rows = np.ones(n_expr, dtype=bool)
        if int(rows.sum()) != len(ey):
            print(f" [WARN] skipping residual panel for {gene}: model/point "
                  f"length mismatch ({len(ey):,} vs {int(rows.sum()):,}).")
            rows = None

        v = v_g[rows] if rows is not None else v_g
        ex, ey_s = (_std_resid(v, model) if rows is not None else (None, None))

        fig, axes = plt.subplots(1, 2, figsize=(3.15, 1.378))
        fig.set_layout_engine("constrained")

        ax = axes[0]
        for age, col in [("young", YOUNG_COL), ("aged", AGED_COL)]:
            m = ages_g == age
            if m.sum() < 2:
                continue
            ax.scatter(y_g[m], v_g[m],
                       facecolors=to_rgba(col, POINT_ALPHA_FILL),
                       edgecolors=to_rgba(col, POINT_ALPHA_EDGE),
                       s=POINT_SIZE, linewidths=POINT_LINEWIDTH,
                       label=age.capitalize(),
                       rasterized=SVG_RASTERIZED_CORRELATION)
            if m.sum() >= 3 and np.ptp(y_g[m]) > 0:
                z = np.polyfit(y_g[m], v_g[m], 1)
                xs = np.linspace(y_g[m].min(), y_g[m].max(), 50)
                ax.plot(xs, np.polyval(z, xs), color=col, lw=1.5, alpha=0.85)

        lines = []
        if stats_by_gene is not None and gene in stats_by_gene.index:
            row = stats_by_gene.loc[gene]
            for nm, lab in [("young", "young"), ("aged", "aged")]:
                r, f = row.get(f"rho_{nm}", np.nan), row.get(f"fdr_{nm}", np.nan)
                if np.isfinite(r):
                    lines.append((lab, f"{lab} {r:+.2f}",
                                  YOUNG_COL if nm == "young" else AGED_COL))
        if lines:
            # Same bug the corrected panel had: a fixed 0.06 step of the axes
            # height is about 2.4 pt, well under the 6 pt glyphs, so "young"
            # and "aged" printed on top of each other and on the points. The
            # step is now derived from the font and the measured axes height,
            # and an empty band is reserved above the data to hold them.
            fig.canvas.draw()
            _h_in = ax.get_window_extent().height / fig.dpi
            _step = (RAW_LABEL_FONT * 1.35 / 72.0) / max(_h_in, 1e-6)
            _lo, _hi = ax.get_ylim()
            ax.set_ylim(_lo, _hi + (_hi - _lo) * (_step * len(lines) + 0.04))
            for i, (_, txt, col) in enumerate(lines):
                ax.text(0.04, 0.985 - _step * i, txt,
                        transform=ax.transAxes,
                        fontsize=RAW_LABEL_FONT, color=col, va="top")

        ax.set_xlabel(axis_target_label(target_col))
        ax.set_ylabel(gene)
        if PANEL_SUBTITLES:
            # "unadjusted" rather than "raw": the lines here are OLS on raw
            # values and do NOT correspond to the depth-adjusted rho quoted
            # beside them. Saying so stops the two being read as the same fit.
            ax.set_title("raw (unadjusted fit)", fontsize=SUBTITLE_FONT)
        if PANEL_LEGEND:
            ax.legend(frameon=False, fontsize=LEGEND_FONT)
        ax.xaxis.set_major_locator(MaxNLocator(VOLCANO_MAX_TICKS))
        ax.yaxis.set_major_locator(MaxNLocator(VOLCANO_MAX_TICKS))
        despine(ax)

        ax = axes[1]
        if ex is not None:
            sub_ages = ages_g[rows]
            for age, col in [("young", YOUNG_COL), ("aged", AGED_COL)]:
                m = sub_ages == age
                if m.sum() < 2:
                    continue
                ax.scatter(ey_s[m], ex[m],
                           facecolors=to_rgba(col, POINT_ALPHA_FILL),
                           edgecolors=to_rgba(col, POINT_ALPHA_EDGE),
                           s=POINT_SIZE, linewidths=POINT_LINEWIDTH,
                           label=age.capitalize(),
                           rasterized=SVG_RASTERIZED_CORRELATION)

            # Reference lines through the origin at slope = rho. On
            # standardised rank residuals that slope IS the correlation, so
            # these are the quoted numbers drawn, not fits to the cloud.
            xs = np.linspace(float(np.nanmin(ey_s)), float(np.nanmax(ey_s)), 50)
            row_r = (stats_by_gene.loc[gene]
                     if stats_by_gene is not None
                     and gene in stats_by_gene.index else None)
            drawn = []
            if row_r is not None:
                r_sel = row_r.get(f"rho_{stat}", np.nan)
                if np.isfinite(r_sel):
                    ax.plot(xs, r_sel * xs, color="black", lw=RHO_LINE_LW, alpha=0.95,
                            zorder=6)
                    drawn.append(f"{stat} {r_sel:+.2f}")
                if RHO_LINES_PER_STRATUM and stat == "global":
                    for nm, c in (("young", YOUNG_COL), ("aged", AGED_COL)):
                        rr = row_r.get(f"rho_{nm}", np.nan)
                        if np.isfinite(rr):
                            ax.plot(xs, rr * xs, color=c, lw=RHO_LINE_LW_STRATUM, alpha=0.9,
                                    ls="--", zorder=5)
                            drawn.append(f"{nm} {rr:+.2f}")
            if drawn:
                ax.text(0.04, 0.96, "\n".join(drawn), transform=ax.transAxes,
                        fontsize=6, va="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="0.7", alpha=0.85))
        else:
            ax.text(0.5, 0.5, "residuals unavailable", ha="center",
                    va="center", transform=ax.transAxes, fontsize=6,
                    color="0.4")

        ax.axhline(0, color="grey", ls="--", lw=0.6)
        ax.axvline(0, color="grey", ls="--", lw=0.6)
        ax.set_xlabel("stiffness resid. (SD)")
        ax.set_ylabel(f"{gene} resid. (SD)")
        if PANEL_SUBTITLES:
            ax.set_title(f"added-variable, slope = rho_{stat}",
                         fontsize=SUBTITLE_FONT)
        if PANEL_LEGEND:
            ax.legend(frameon=False, fontsize=LEGEND_FONT)
        ax.xaxis.set_major_locator(MaxNLocator(VOLCANO_MAX_TICKS))
        ax.yaxis.set_major_locator(MaxNLocator(VOLCANO_MAX_TICKS))
        despine(ax)

        fig.suptitle(f"{group_disp(group_label)} \u00b7 {_italic(gene)} "
                     f"\u00b7 {stat_disp(stat)}", fontsize=TITLE_FONT)
        save_figure(fig, make_plot_stem(3, f"{group_label}{stem_tag}",
                                        "correlation", target_col,
                                        gene=gene), out_fig)
        plt.close(fig)

        # The readable companion: same axes as the raw panel, depth removed.
        if PLOT_ADJUSTED_PANEL and log_depth_g is not None:
            v_adj = depth_adjusted_expression(v_g, log_depth_g)
            plot_adjusted_correlation(
                y_g, v_adj, ages_g, gene, target_col, out_fig,
                group_label=group_label, stat=stat, stem_tag=stem_tag,
                row=row_g)

        plotted += 1

    if PLOT_RAW_PANEL:
        print(f" Plotted {plotted} two-panel figures"
              + (" (plus depth-adjusted companions)" if PLOT_ADJUSTED_PANEL
                 and log_depth_all is not None else "")
              + (f"; skipped {skipped} gene(s) below min subset size."
                 if skipped else "."))
    else:
        print(f" Plotted {plotted} corrected panels "
              f"(PLOT_RAW_PANEL=False, so no raw/residual figure)"
              + (f"; skipped {skipped} gene(s) below min subset size."
                 if skipped else "."))

# =============================================================================
# STEP 7: PATHWAY ENRICHMENT (GSEA PRERANK ONLY; ORA REMOVED)
# =============================================================================

def get_prerank_gene_sets(mechanomics=None):
    gene_sets = {}
    for gs_name in GSEA_GENESETS:
        try:
            gs_dict = gp.get_library(name=gs_name, organism=GSEA_ORGANISM)
            gene_sets.update(gs_dict)
            print(f" Loaded gene set library: {gs_name} ({len(gs_dict):,} terms)")
        except Exception as e:
            print(f" WARNING: failed to load gene set library {gs_name}: {e}")
    if mechanomics:
        gene_sets["Mechanomics"] = sorted(set(mechanomics))
        print(f" Added custom term 'Mechanomics' "
              f"({len(gene_sets['Mechanomics']):,} genes)")
    return gene_sets

def plot_pathway_dotplot(res_df, out_dir, stem, title,
                         top_n=PATHWAY_DOTPLOT_TOP_N,
                         fdr_col="FDR q-val", term_col="Term", score_col="NES",
                         size_col="Tag %", xlabel=None, cbar_label=None):
    if res_df is None or len(res_df) == 0:
        print(f" No pathway results for {stem}; skipping dot plot.")
        return

    d = res_df.copy()
    for c in (fdr_col, score_col):
        if c not in d.columns:
            print(f" WARNING: column '{c}' missing for {stem}; skipping dot plot.")
            return
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[fdr_col, score_col, term_col])

    n_all = len(d)
    d = d[d[fdr_col] < PATHWAY_DOTPLOT_FDR_MAX]
    if d.empty:
        print(f" No terms below FDR {PATHWAY_DOTPLOT_FDR_MAX} for {stem}.")
        return
    n_sig = len(d)
    # Balanced by direction: half the slots to the most positive NES, half to
    # the most negative. Ranking by |NES| alone can return 15 positives and no
    # negatives, which hides exactly the contrast the plot is meant to show.
    # top_n is a TOTAL and is halved, so 30 gives 15 up and 15 down.
    half = max(int(top_n) // 2, 1)
    pos = d[d[score_col] > 0].nlargest(half, score_col)
    neg = d[d[score_col] < 0].nsmallest(half, score_col)
    n_pos_avail = int((d[score_col] > 0).sum())
    n_neg_avail = int((d[score_col] < 0).sum())
    d = pd.concat([neg, pos]).sort_values(score_col)
    print(f" {n_sig} of {n_all} terms passed FDR < {PATHWAY_DOTPLOT_FDR_MAX} "
          f"({n_pos_avail} positive, {n_neg_avail} negative); "
          f"drawing {len(pos)} positive + {len(neg)} negative"
          + ("  [TRUNCATED, see the CSV for the rest]" if n_sig > len(d) else ""))
    if len(pos) < half or len(neg) < half:
        print(f"   NOTE: fewer than {half} term(s) available in one direction; "
              f"the panel is asymmetric by data, not by choice.")
    if d.empty:
        print(" Nothing left to draw after the directional split.")
        return

    # ---- bubble size: absolute bins, identical on every panel ---------------
    # gseapy writes "Tag %" as "<leading edge hits>/<gene set size>", e.g.
    # "11/29". Verified against a live gp.prerank run: the denominator is the
    # gene set size after matching to the ranked list, which is what the key
    # calls "genes in set".
    sz = None
    if size_col and size_col in d.columns:
        sz = pd.to_numeric(
            d[size_col].astype(str).str.extract(r"/\s*(\d+)")[0], errors="coerce")
        if sz.isna().all():
            sz = pd.to_numeric(
                d[size_col].astype(str).str.replace("%", "", regex=False),
                errors="coerce")
    if sz is None or sz.isna().all():
        print("   NOTE: no usable gene-set size column; every bubble is drawn "
              "at the middle size.")
        sz = pd.Series(np.full(len(d), PATHWAY_DOTPLOT_SIZE_BREAKS[0]),
                       index=d.index)
    sz = sz.fillna(float(PATHWAY_DOTPLOT_SIZE_BREAKS[0]))

    b_lo, b_hi = PATHWAY_DOTPLOT_SIZE_BREAKS
    a_s, a_m, a_l = PATHWAY_DOTPLOT_SIZE_AREAS

    def size_area(v):
        v = np.asarray(v, float)
        return np.where(v < b_lo, a_s, np.where(v < b_hi, a_m, a_l))

    n_small = int((sz < b_lo).sum())
    n_med = int(((sz >= b_lo) & (sz < b_hi)).sum())
    n_large = int((sz >= b_hi).sum())
    print(f"   Gene-set size bins: {n_small} small (< {b_lo}), "
          f"{n_med} medium ({b_lo} to {b_hi - 1}), {n_large} large (>= {b_hi})")

    nl = -np.log10(pd.to_numeric(d[fdr_col], errors="coerce")
                   .clip(lower=1e-300))
    capped = nl >= PATHWAY_DOTPLOT_CMAX
    if PATHWAY_DOTPLOT_MARK_CAPPED:
        labels = [f"{str(t)[:PATHWAY_DOTPLOT_TERM_CHARS]}{' *' if c else ''}"
                  for t, c in zip(d[term_col], capped)]
    else:
        labels = [str(t)[:PATHWAY_DOTPLOT_TERM_CHARS] for t in d[term_col]]
    ypos = np.arange(len(d))

    # ---- figure and explicit rectangles -------------------------------------
    # Width identical for every GSEA panel; height free between a floor and a
    # ceiling so a 30-term panel gets the room it needs and a 4-term panel is
    # not stretched to match it.
    fig_h_mm = PATHWAY_DOTPLOT_PAD_MM + PATHWAY_DOTPLOT_ROW_MM * len(d)
    fig_h_mm = float(np.clip(fig_h_mm, PATHWAY_DOTPLOT_H_MIN_MM,
                             PATHWAY_DOTPLOT_H_MAX_MM))
    W = float(PATHWAY_DOTPLOT_W_MM)
    fig = plt.figure(figsize=(W / 25.4, fig_h_mm / 25.4))
    fig.set_layout_engine("none")

    ax_rect = (PATHWAY_DOTPLOT_LEFT_MM / W,
               PATHWAY_DOTPLOT_BOTTOM_MM / fig_h_mm,
               (W - PATHWAY_DOTPLOT_LEFT_MM - PATHWAY_DOTPLOT_RIGHT_MM) / W,
               1.0 - (PATHWAY_DOTPLOT_BOTTOM_MM + PATHWAY_DOTPLOT_TOP_MM)
               / fig_h_mm)
    ax = fig.add_axes(ax_rect)

    x_col = (W - PATHWAY_DOTPLOT_RIGHT_MM + PATHWAY_DOTPLOT_GAP_MM) / W
    pad = PATHWAY_DOTPLOT_COL_PAD_FRAC
    split = PATHWAY_DOTPLOT_SPLIT_FRAC
    # Colour bar in the top half of the figure, bubble key in the bottom half,
    # both in the same right-hand column. The pad keeps the ">= 5" tick label
    # off the top edge of the canvas, which has no margin (bbox_inches=None).
    cax = fig.add_axes((x_col, split + pad,
                        PATHWAY_DOTPLOT_CBAR_W_MM / W,
                        (1.0 - split) - 3.0 * pad))
    lax = fig.add_axes((x_col, pad,
                        (PATHWAY_DOTPLOT_RIGHT_MM - PATHWAY_DOTPLOT_GAP_MM) / W,
                        split - 2.0 * pad))
    lax.set_axis_off()

    ax.axvline(0, ls="--", color="grey", lw=0.6, zorder=1)
    sc = ax.scatter(d[score_col], ypos,
                    c=nl.clip(lower=PATHWAY_DOTPLOT_CMIN,
                              upper=PATHWAY_DOTPLOT_CMAX),
                    s=size_area(sz), cmap="Reds",
                    vmin=PATHWAY_DOTPLOT_CMIN, vmax=PATHWAY_DOTPLOT_CMAX,
                    edgecolors="black", linewidths=0.45, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_ylim(-0.75, len(d) - 0.25)
    if PATHWAY_DOTPLOT_SYMMETRIC_X:
        # Zero centred. Half-width is the largest |NES| on the panel, so the
        # dashed zero line sits exactly in the middle whichever side the terms
        # fall on.
        half = float(np.nanmax(np.abs(d[score_col].values))) * 1.12
        half = max(half, 0.5)
        ax.set_xlim(-half, half)
    ax.set_xlabel(xlabel or score_col, fontsize=6)
    ax.tick_params(axis="x", labelsize=6)
    # Title on the FIGURE, not the axes. An axes title is centred on the axes,
    # which sit left of the colour bar column, so a long title ran off the
    # right-hand side and into the bar.
    fig.text(0.5, 1.0 - (PATHWAY_DOTPLOT_TOP_MM * 0.42) / fig_h_mm, title,
             ha="center", va="top", fontsize=6)
    despine(ax)

    cb = fig.colorbar(sc, cax=cax)
    cb.set_ticks([PATHWAY_DOTPLOT_CMIN, PATHWAY_DOTPLOT_CMAX])
    cb.set_ticklabels([f"{PATHWAY_DOTPLOT_CMIN:g}",
                       f">= {PATHWAY_DOTPLOT_CMAX:g}"])
    cb.set_label(cbar_label or r"$-\log_{10}$(FDR)", fontsize=6)
    cb.ax.tick_params(labelsize=6)

    handles = [lax.scatter([], [], s=a, facecolor="lightgrey",
                           edgecolor="black", linewidths=0.45, label=lab)
               for a, lab in zip(PATHWAY_DOTPLOT_SIZE_AREAS,
                                 PATHWAY_DOTPLOT_SIZE_LABELS)]
    lax.legend(handles=handles, title="Genes in set", frameon=False,
               labelspacing=1.4, loc="center left", borderpad=0.0,
               handletextpad=0.8, fontsize=6, title_fontsize=6)

    save_figure(fig, stem, out_dir)
    plt.close(fig)

def run_pathway_analysis(res, out_dir, target_col, group_label="global",
                         mechanomics=None, prerank_gene_sets=None,
                         rank_col=RANK_COL_FOR_PATHWAYS):
    """
    Preranked GSEA on the FULL ranked gene list, ranked by rank_col.

    GSEA is used rather than ORA because it weighs every tested gene by its
    correlation instead of thresholding into an in/out list. With effect sizes
    this small, a hard cut throws away most of the signal, and ORA additionally
    needs a background that Enrichr will not accept.
    """
    if RUN_GSEA and not GSEAPY_AVAILABLE:
        print(f" [SKIP] Pathway analysis for {group_label}: gseapy is not "
              f"importable ({GSEAPY_IMPORT_ERROR}).")
        print(f"        Everything else in this run is unaffected. To enable "
              f"GSEA:  pip install gseapy")
        return None
    if not RUN_GSEA:
        print(f" RUN_GSEA=False; skipping pathway analysis for {group_label}.")
        return None

    print("\n" + "=" * 60)
    print(f"PATHWAY ANALYSIS (GSEA prerank): {group_label}, {target_col}")
    print(f"Ranked by: {rank_col}")
    print("=" * 60)

    pathway_dir = out_dir / f"{group_label}_{target_col}_pathway_analysis"
    gsea_dir = pathway_dir / "gsea_prerank"
    gsea_dir.mkdir(parents=True, exist_ok=True)

    if rank_col not in res.columns:
        print(f" No column {rank_col}; skipping.")
        return None

    # ---- build the ranking vector -----------------------------------------
    # GSEA_RANK_METRIC decides what plays the part of log2FoldChange. See the
    # config block for why "rho" is the default.
    stat_name = str(rank_col).replace("rho_", "")
    fdr_col = f"fdr_{stat_name}"
    rnk = res[["gene", rank_col]].copy()
    metric = GSEA_RANK_METRIC

    if metric in ("signed_neglog10_fdr", "rho_x_neglog10_fdr"):
        if fdr_col not in res.columns:
            print(f" [NOTE] GSEA_RANK_METRIC = '{metric}' needs {fdr_col}, "
                  f"which is not in the results. Falling back to 'rho'.")
            metric = "rho"
        else:
            rnk[fdr_col] = pd.to_numeric(res[fdr_col], errors="coerce")

    if metric == "rho":
        rnk["rank_metric"] = pd.to_numeric(rnk[rank_col], errors="coerce")
    else:
        nlq = -np.log10(rnk[fdr_col].clip(lower=GSEA_RANK_FDR_FLOOR))
        r_ = pd.to_numeric(rnk[rank_col], errors="coerce")
        rnk["rank_metric"] = (np.sign(r_) * nlq if metric == "signed_neglog10_fdr"
                              else r_ * nlq)

    rnk = (rnk[["gene", "rank_metric"]].dropna()
           .sort_values("rank_metric", ascending=False)
           .drop_duplicates(subset="gene"))
    print(f" Ranking metric: {metric}"
          + ("" if metric == GSEA_RANK_METRIC
             else f" (requested '{GSEA_RANK_METRIC}')"))
    n_zero = int((rnk["rank_metric"] == 0).sum())
    if n_zero:
        print(f"   {n_zero:,} gene(s) rank exactly 0; GSEA places these in the "
              f"middle of the list, which is correct but means they carry no "
              f"directional weight.")
    if len(rnk) < 200:
        print(f" Only {len(rnk)} ranked genes; GSEA would not be meaningful. "
              f"Skipping.")
        return None

    rnk.to_csv(pathway_dir / f"{group_label}_{target_col}_{rank_col}_ranked_genes.rnk",
               sep="\t", index=False, header=False)
    print(f" Ranked list: {len(rnk):,} genes, "
          f"{rnk['rank_metric'].min():+.3f} to {rnk['rank_metric'].max():+.3f}")

    if prerank_gene_sets is None:
        prerank_gene_sets = get_prerank_gene_sets(mechanomics)
    if not prerank_gene_sets:
        print(" No gene sets could be loaded; skipping GSEA.")
        return None

    try:
        pre = gp.prerank(
            rnk=rnk,
            gene_sets=prerank_gene_sets,
            outdir=str(gsea_dir),
            permutation_num=GSEA_PERM,
            seed=RANDOM_STATE,
            min_size=GSEA_MIN_SIZE,
            max_size=GSEA_MAX_SIZE,
            verbose=False,
        )
    except Exception as e:
        print(f" WARNING: preranked GSEA failed for {group_label}: {e}")
        return None

    if getattr(pre, "res2d", None) is None:
        print(" GSEA returned no results.")
        return None

    gsea_res = pre.res2d.copy()
    gsea_res.to_csv(
        pathway_dir / f"{group_label}_{target_col}_{rank_col}_prerank_gsea_results.csv",
        index=False)
    n_sig = int((pd.to_numeric(gsea_res["FDR q-val"], errors="coerce")
                 < PATHWAY_DOTPLOT_FDR_MAX).sum())
    print(f" Preranked GSEA: {n_sig} of {len(gsea_res)} terms at "
          f"FDR < {PATHWAY_DOTPLOT_FDR_MAX}")

    mech_row = gsea_res[gsea_res["Term"].astype(str).str.contains(
        "Mechanomics", case=False, na=False)]
    if len(mech_row):
        r = mech_row.iloc[0]
        print(f" Mechanomics term: NES={r.get('NES')}, "
              f"FDR={r.get('FDR q-val')}")

    plot_pathway_dotplot(
        gsea_res, pathway_dir,
        make_plot_stem(4, group_label, f"pathway_dotplot_prerank_{rank_col}",
                       target_col),
        title=f"{group_label}: {target_col} preranked GSEA ({rank_col})",
        top_n=2 * PATHWAY_DOTPLOT_TOP_N_PER_DIRECTION,
        fdr_col="FDR q-val", term_col="Term", score_col="NES", size_col="Tag %")
    return gsea_res

# =============================================================================
# STEP 7b: OVER-REPRESENTATION ANALYSIS (ORA)
#
# Replaces the whole STEP 7b block. Paste over the three existing functions
# (_ora_filter_gene_sets, _ora_signed_score, run_ora_analysis).
#
# WHY THE FIRST VERSION PRODUCED NOTHING
# Every comparison logged "Gene sets: 0 of 7,158 ... testable here". Zero of
# 7,158 is not a size filter doing its job, it is a key mismatch: the two sides
# were never comparing the same strings.
#
# Enrichr's libraries store gene symbols in UPPER CASE (COL1A1), and the
# expression matrix here uses mouse case (Col1a1). gp.prerank handles that
# internally: gseapy/base.py samples up to 20 gene sets, and if they are all
# uppercase while the query is not, it uppercases the query before matching
# (gseapy/enrichr.py does the same for gp.enrich, including the background).
# That is why GSEA on the SAME 7,158 sets returned real terms while the plain
# set.intersection() in _ora_filter_gene_sets matched nothing at all.
#
# The fix is to do every comparison on a case-folded key, and to keep a map
# back to the original symbols so the reported gene lists still read as mouse
# symbols. Everything handed to gp.enrich (sets, query, background) is now in
# one consistent case, so gseapy's own guesswork never has to fire.
# =============================================================================

def _ora_key(g):
    """Case-folded, whitespace-stripped matching key for a gene symbol."""
    return str(g).strip().upper()


def _ora_filter_gene_sets(gene_sets, background):
    """
    Intersect every gene set with the background on a case-folded key, then
    keep the sets whose SURVIVING size is within the bounds.

    Returns (sets_out, overlap_sizes, key_to_symbol) where
      sets_out       term -> list of UPPERCASE keys present in the background
      overlap_sizes  the overlap size of every input set, for diagnostics
      key_to_symbol  UPPERCASE key -> the original symbol from the background

    Size filtering is done here rather than left to gseapy because gp.enrich
    has no min_size or max_size argument, and because a set whose genes are
    mostly absent from this experiment should be judged on the genes that are
    actually testable.
    """
    key_to_symbol = {}
    for g in background:
        key_to_symbol.setdefault(_ora_key(g), g)

    sets_out, overlap_sizes = {}, []
    for term, genes in gene_sets.items():
        keys = {_ora_key(x) for x in genes}
        hit = sorted(k for k in keys if k in key_to_symbol)
        overlap_sizes.append(len(hit))
        if ORA_MIN_SET_SIZE <= len(hit) <= ORA_MAX_SET_SIZE:
            sets_out[term] = hit
    return sets_out, overlap_sizes, key_to_symbol


def _ora_report_no_sets(gene_sets, background, overlap_sizes):
    """
    Say WHY nothing survived, with the evidence on screen.

    The previous version printed one line and moved on, so a total key
    mismatch and a legitimately strict size filter looked identical.
    """
    ov = np.asarray(overlap_sizes, dtype=float) if len(overlap_sizes) else \
        np.array([0.0])
    n_zero = int((ov == 0).sum())
    print(f"   Overlap with the background, across {len(ov):,} gene sets: "
          f"min {ov.min():.0f}, median {np.median(ov):.0f}, max {ov.max():.0f}; "
          f"{n_zero:,} set(s) overlap by zero genes.")
    if n_zero == len(ov):
        some_term = next(iter(gene_sets)) if gene_sets else None
        sample_set = list(gene_sets[some_term])[:6] if some_term else []
        sample_bg = list(background)[:6]
        print("   EVERY set overlaps by zero. That is a symbol mismatch, not a "
              "threshold. Compare:")
        print(f"     gene set '{str(some_term)[:40]}': {sample_set}")
        print(f"     background:                      {sample_bg}")
        print("   If one side is upper case and the other is not, the case "
              "folding in _ora_filter_gene_sets has been bypassed or the "
              "species is wrong.")
    else:
        print(f"   Sets do overlap the background, but none lands inside "
              f"ORA_MIN_SET_SIZE = {ORA_MIN_SET_SIZE} to ORA_MAX_SET_SIZE = "
              f"{ORA_MAX_SET_SIZE}. Widen those if that is not what you want.")


def _ora_report_size_bands(overlap_sizes):
    """
    How many gene sets sit in each size band AFTER intersecting with the
    background. Printed on every ORA run so the size bounds can be chosen from
    this experiment's own numbers rather than from a guess about what a wider
    bound might add.
    """
    ov = np.asarray(overlap_sizes, dtype=int) if len(overlap_sizes) else \
        np.array([0], dtype=int)
    lo, hi = ORA_MIN_SET_SIZE, ORA_MAX_SET_SIZE
    bands = [("0 (no overlap)", ov == 0),
             (f"1 to {lo - 1} (below ORA_MIN_SET_SIZE)",
              (ov > 0) & (ov < lo)),
             (f"{lo} to {hi} (TESTED)", (ov >= lo) & (ov <= hi))]
    for upper in ORA_SIZE_REPORT_BANDS:
        if upper <= hi:
            continue
        prev = max(hi, max((u for u in ORA_SIZE_REPORT_BANDS if u < upper),
                           default=hi))
        bands.append((f"{prev + 1} to {upper}", (ov > prev) & (ov <= upper)))
    top = max([hi] + [u for u in ORA_SIZE_REPORT_BANDS if u > hi])
    bands.append((f"> {top}", ov > top))

    print("   Gene-set sizes after intersecting with the background:")
    for label, mask in bands:
        n = int(mask.sum())
        if n:
            print(f"     {label:38s} {n:6,d} set(s)")
    for upper in ORA_SIZE_REPORT_BANDS:
        if upper > hi:
            extra = int(((ov > hi) & (ov <= upper)).sum())
            print(f"   Raising ORA_MAX_SET_SIZE from {hi} to {upper} would add "
                  f"{extra:,} testable set(s) ({extra / max(len(ov), 1):.1%} "
                  f"more of the library). Broader sets, more tests.")


def _ora_signed_score(df, direction):
    """
    log2(odds ratio), signed by which gene list the term came from.

    The odds ratio is +inf when every background gene in the set is a hit, so
    it is clipped. Returns NaN where the odds ratio is not positive, which
    happens when the overlap is zero.
    """
    orr = pd.to_numeric(df["Odds Ratio"], errors="coerce")
    orr = orr.replace([np.inf, -np.inf], np.nan).clip(upper=ORA_OR_CLIP)
    val = np.log2(orr.where(orr > 0))
    return val if direction == "positive" else -val


def _ora_genes_to_symbols(cell, key_to_symbol):
    """Map gseapy's semicolon-joined UPPERCASE hits back to mouse symbols."""
    if not isinstance(cell, str) or not cell:
        return cell
    return ";".join(key_to_symbol.get(_ora_key(g), g) for g in cell.split(";"))


def run_ora_analysis(res, out_dir, target_col, group_label="global",
                     stat="global", gene_sets=None, mechanomics=None):
    """
    Hypergeometric over-representation of the SIGNIFICANT genes, run separately
    for the positively and negatively correlated lists.

    The gate is identical to the one used everywhere else in this script
    (fdr_<stat> < FDR_THRESH and |rho_<stat>| >= RHO_THRESH), so the ORA input
    is exactly the gene list in <group>_<target>_significant_only_r_values.csv.
    The background is every gene with a finite rho_<stat> in this group, which
    is the set of genes that had a chance to be a hit.

    Returns a DataFrame of all terms from both directions, or None.
    """
    if not RUN_ORA:
        return None
    if not GSEAPY_AVAILABLE:
        print(f" [SKIP] ORA for {group_label}/{stat}: gseapy is not importable "
              f"({GSEAPY_IMPORT_ERROR}).")
        return None

    rho_col, fdr_col = f"rho_{stat}", f"fdr_{stat}"
    if rho_col not in res.columns or fdr_col not in res.columns:
        print(f" [SKIP] ORA for {group_label}/{stat}: no {rho_col}/{fdr_col}.")
        return None

    d = res[["gene", rho_col, fdr_col]].copy()
    d[rho_col] = pd.to_numeric(d[rho_col], errors="coerce")
    d[fdr_col] = pd.to_numeric(d[fdr_col], errors="coerce")
    d = d.dropna(subset=["gene", rho_col])

    background = sorted(set(d["gene"].astype(str)))
    sig = d[(d[fdr_col] < FDR_THRESH) & (d[rho_col].abs() >= RHO_THRESH)]
    lists = {
        "positive": sorted(set(sig.loc[sig[rho_col] > 0, "gene"].astype(str))),
        "negative": sorted(set(sig.loc[sig[rho_col] < 0, "gene"].astype(str))),
    }

    print("\n" + "=" * 60)
    print(f"ORA (hypergeometric): {group_label}, {target_col}, {stat}")
    print("=" * 60)
    print(f" Background: {len(background):,} genes with a finite {rho_col} "
          f"(NOT the whole genome; see the RUN_ORA config note).")
    print(f" Gate: FDR < {FDR_THRESH} and |rho| >= {RHO_THRESH}"
          f"  ->  {len(lists['positive']):,} positive, "
          f"{len(lists['negative']):,} negative")

    if gene_sets is None:
        gene_sets = get_prerank_gene_sets(mechanomics)
    if not gene_sets:
        print(" No gene sets available; ORA skipped.")
        return None

    sets_bg, overlap_sizes, key_to_symbol = _ora_filter_gene_sets(
        gene_sets, background)
    print(f" Gene sets: {len(sets_bg):,} of {len(gene_sets):,} have "
          f"{ORA_MIN_SET_SIZE} to {ORA_MAX_SET_SIZE} genes present in the "
          f"background and are testable here.")
    _ora_report_size_bands(overlap_sizes)
    if not sets_bg:
        _ora_report_no_sets(gene_sets, background, overlap_sizes)
        print(" No gene set survives the background intersection; ORA skipped.")
        return None

    # Everything is matched on the same case-folded key, so gseapy never has to
    # guess. bg_keys is the background in that key space.
    bg_keys = sorted(key_to_symbol)

    ora_dir = out_dir / f"{group_label}_{target_col}_pathway_analysis" / "ora"
    ora_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for direction in ORA_DIRECTIONS:
        genes = lists.get(direction, [])
        if len(genes) < ORA_MIN_GENES:
            print(f" [SKIP] {direction}: {len(genes)} gene(s), below "
                  f"ORA_MIN_GENES = {ORA_MIN_GENES}. A list this short can "
                  f"only produce terms driven by one or two genes, so it is "
                  f"not tested rather than reported as 'nothing found'.")
            continue

        query_keys = sorted({_ora_key(g) for g in genes})
        n_in_bg = sum(1 for k in query_keys if k in key_to_symbol)
        if n_in_bg < len(query_keys):
            print(f"   NOTE: {len(query_keys) - n_in_bg} of {len(query_keys)} "
                  f"{direction} gene(s) are not in the background. That should "
                  f"be impossible, since both come from the same table.")

        try:
            en = gp.enrich(gene_list=query_keys, gene_sets=sets_bg,
                           background=bg_keys, outdir=None,
                           no_plot=True, verbose=False)
        except Exception as e:
            print(f" WARNING: ORA failed for {group_label}/{stat}/{direction}: "
                  f"{e}")
            continue
        r = getattr(en, "results", None)
        if r is None or not len(r):
            print(f" {direction}: gseapy returned no rows.")
            continue

        r = r.copy()
        if "Genes" in r.columns:
            r["Genes"] = r["Genes"].map(
                lambda c: _ora_genes_to_symbols(c, key_to_symbol))
        r["direction"] = direction
        r["group"] = group_label
        r["stat"] = stat
        r["n_query_genes"] = len(query_keys)
        r["n_background"] = len(bg_keys)
        r["signed_log2_or"] = _ora_signed_score(r, direction)
        r = r.sort_values("Adjusted P-value")
        r.to_csv(ora_dir / f"{group_label}_{target_col}_{stat}_{direction}"
                           f"_ora_results.csv", index=False)

        n_sig = int((pd.to_numeric(r["Adjusted P-value"], errors="coerce")
                     < ORA_FDR_MAX).sum())
        print(f" {direction}: {len(query_keys):,} genes -> {n_sig} of {len(r)} "
              f"terms at BH FDR < {ORA_FDR_MAX}")
        if n_sig:
            for _, row in r.head(3).iterrows():
                print(f"    {str(row['Term'])[:55]:57s} "
                      f"overlap {row['Overlap']:>8}  "
                      f"OR {float(row['Odds Ratio']):6.2f}  "
                      f"FDR {float(row['Adjusted P-value']):.2e}")
        frames.append(r)

    if not frames:
        print(" ORA produced nothing for this comparison.")
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(
        ora_dir / f"{group_label}_{target_col}_{stat}_ora_results_combined.csv",
        index=False)

    if ORA_COMBINED_DOTPLOT:
        plot_df = combined
        if ORA_PLOT_ENRICHED_ONLY:
            orr = pd.to_numeric(combined["Odds Ratio"], errors="coerce")
            keep = orr > 1.0
            n_drop = int((~keep & (pd.to_numeric(combined["Adjusted P-value"],
                                                 errors="coerce")
                                   < ORA_FDR_MAX)).sum())
            if n_drop:
                print(f" {n_drop} term(s) below FDR {ORA_FDR_MAX} have an odds "
                      f"ratio < 1 (UNDER-represented) and are left off the "
                      f"figure; they are in the CSV.")
            plot_df = combined[keep]
        plot_pathway_dotplot(
            plot_df, ora_dir,
            make_plot_stem(4, group_label, f"ora_dotplot_{stat}", target_col),
            title=f"{group_label}: {target_col} ORA ({stat}, "
                  f"pos vs neg gene lists)",
            top_n=2 * PATHWAY_DOTPLOT_TOP_N_PER_DIRECTION,
            fdr_col="Adjusted P-value", term_col="Term",
            score_col="signed_log2_or", size_col="Overlap",
            xlabel=r"$\log_2$ odds ratio (+ positive list, - negative list)",
            cbar_label=r"$-\log_{10}$(BH FDR)")
    return combined

# =============================================================================
# STEP 8: AGED VS YOUNG CORRELATION COMPARISON
# =============================================================================

def compare_aged_young(res, out_dir, target_col, group_label="global"):
    """
    rho_young against rho_aged, from the statistics already computed. Genes
    whose difference passes the Fisher z test (FDR) are highlighted, and the
    largest differences are labelled.

    This is where Simpson's paradox shows up: Aldh1a1 was -0.130 pooled but
    +0.018 in young and -0.061 in aged.

    Two CSVs are written:
      <group>_<target>_compare_age_groups.csv
          EVERY gene that has both a young and an aged rho, significant or
          not, plus significant_diff, delta_direction and labelled_in_plot.
      <group>_<target>_compare_age_groups_significant_only.csv
          only the genes coloured in the figure, sorted by |rho_delta|.
    """
    need = ["rho_young", "rho_aged", "rho_delta", "fdr_delta"]
    if any(c not in res.columns for c in need):
        print(" Age-stratified statistics not available; skipping comparison.")
        return None

    d = res[["gene"] + need + ["p_delta"]].dropna(
        subset=["rho_young", "rho_aged"]).copy()
    if d.empty:
        print(" No genes with both young and aged correlations.")
        return None

    d["significant_diff"] = (d["fdr_delta"] < FDR_THRESH) & \
                            (d["rho_delta"].abs() >= RHO_DIFF_THRESH)

    # Which side of the diagonal. rho_delta = rho_aged - rho_young, so a
    # positive delta is the top left of the panel and a negative delta the
    # bottom right. This is the SAME quantity the Fisher z test uses, so the
    # colour and the significance call cannot disagree.
    d["delta_direction"] = np.where(
        ~d["significant_diff"], "ns",
        np.where(d["rho_delta"] > 0, "stronger_in_aged", "stronger_in_young"))

    d = d.sort_values("p_delta")

    n_label = TOP_N_LABEL if DELTA_LABEL_N is None else int(DELTA_LABEL_N)
    # The labelled genes are chosen ONLY from the coloured (significant) set,
    # so nothing unlabelled-but-significant is implied to be a hit and nothing
    # non-significant gets a name.
    sg_all = d[d["significant_diff"]].assign(_a=lambda t: t["rho_delta"].abs())
    top = sg_all.sort_values("_a", ascending=False).head(n_label)
    label_set = set(top["gene"])
    d["labelled_in_plot"] = d["gene"].isin(label_set)

    d.to_csv(out_dir / f"{group_label}_{target_col}_compare_age_groups.csv",
             index=False)
    sig_out = d[d["significant_diff"]].copy()
    sig_out = sig_out.reindex(
        sig_out["rho_delta"].abs().sort_values(ascending=False).index)
    sig_out.to_csv(
        out_dir /
        f"{group_label}_{target_col}_compare_age_groups_significant_only.csv",
        index=False)

    n_diff = int(d["significant_diff"].sum())
    n_up = int((d["delta_direction"] == "stronger_in_aged").sum())
    n_dn = int((d["delta_direction"] == "stronger_in_young").sum())
    n_ns = int(len(d) - n_diff)
    print(f" Differentially correlated with age: {n_diff} of {len(d):,} genes "
          f"(FDR < {FDR_THRESH} and |delta rho| >= {RHO_DIFF_THRESH})")
    print(f"   {n_up} stronger in aged (above the diagonal, {AGED_COL}), "
          f"{n_dn} stronger in young (below it, {YOUNG_COL})")
    if len(top):
        print(f"   {len(top)} labelled on the _top{len(top)} figure; the full "
              f"table and a significant-only table are both written next "
              f"to it.")

    for with_labels in (False, True):
        if with_labels and not len(top):
            # Nothing passed the gate, so the labelled panel would be the
            # unlabelled one under a title promising labels. Not written.
            print("   No gene passed the differential gate; the labelled "
                  "panel is not written.")
            continue
        fig, ax = plt.subplots(figsize=DELTA_FIG_SIZE)
        fig.set_layout_engine("constrained")
        ns = d[~d["significant_diff"]]
        ax.scatter(ns["rho_young"], ns["rho_aged"], color=DELTA_NS_COL,
                   s=DELTA_POINT_SIZE_NS, alpha=0.6, linewidths=0,
                   rasterized=SVG_RASTERIZED_SCATTER)
        if DELTA_SPLIT_COLOURS:
            groups = [("stronger_in_aged", AGED_COL),
                      ("stronger_in_young", YOUNG_COL)]
        else:
            groups = [("stronger_in_aged", DELTA_SIG_COL),
                      ("stronger_in_young", DELTA_SIG_COL)]
        for key, col in groups:
            sub = d[d["delta_direction"] == key]
            if not len(sub):
                continue
            ax.scatter(sub["rho_young"], sub["rho_aged"], color=col,
                       s=DELTA_POINT_SIZE_SIG, alpha=0.85, linewidths=0,
                       rasterized=SVG_RASTERIZED_SCATTER, zorder=3)

        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "--", color="0.45", alpha=0.8, lw=0.6, zorder=1)
        ax.axhline(0, color="grey", lw=0.4, ls=":", zorder=1)
        ax.axvline(0, color="grey", lw=0.4, ls=":", zorder=1)

        show_key = not (with_labels and not DELTA_KEY_ON_LABELLED)

        # Reserve blank space BEFORE anything else needs the limits, so the
        # key never lands on the cloud and adjust_text has room to work.
        pad_top = DELTA_KEY_HEADROOM if show_key else 0.0
        if with_labels:
            lo, hi = ax.get_xlim()
            pad = (hi - lo) * DELTA_LABEL_MARGIN / 2.0
            ax.set_xlim(lo - pad, hi + pad)
            pad_top = max(pad_top, DELTA_LABEL_MARGIN / 2.0)
        lo, hi = ax.get_ylim()
        rng = hi - lo
        ax.set_ylim(lo - (rng * DELTA_LABEL_MARGIN / 2.0 if with_labels else 0),
                    hi + rng * pad_top)

        title = f"{nice_target_label(target_col)}: Aged vs Young"
        n_shown = 0
        if with_labels:
            n_shown = len(top)
            texts = []
            for _, row in top.iterrows():
                t = ax.text(row["rho_young"], row["rho_aged"], row["gene"],
                            fontsize=DELTA_GENE_LABEL_FONT,
                            fontweight="bold", zorder=6)
                t.set_clip_on(True)
                texts.append(t)
                ax.scatter([row["rho_young"]], [row["rho_aged"]],
                           facecolors="none", edgecolors="black",
                           s=DELTA_LABEL_RING_SIZE,
                           linewidths=DELTA_LABEL_RING_LW, zorder=5)
            if texts:
                arrow = dict(arrowstyle="-", color="black", lw=0.3, alpha=0.7)
                try:
                    # adjustText >= 1.0
                    adjust_text(texts, ax=ax, arrowprops=arrow,
                                expand=(1.15, 1.3),
                                force_text=(0.4, 0.7),
                                force_static=(0.2, 0.4),
                                ensure_inside_axes=True)
                except TypeError:
                    # adjustText 0.8 keyword names
                    adjust_text(texts, ax=ax, arrowprops=arrow,
                                expand_points=(1.2, 1.4),
                                expand_text=(1.2, 1.4),
                                force_text=(0.5, 0.8),
                                force_points=(0.3, 0.5))
            title += "\n" + rf"top {n_shown} by $|\Delta r|$"

        # Wrap each explicit line separately so the manual break survives.
        title = "\n".join("\n".join(textwrap.wrap(ln, DELTA_TITLE_WRAP)) or ln
                          for ln in title.split("\n"))
        ax.set_title(title, fontsize=DELTA_TITLE_FONT, linespacing=1.1)

        ax.set_xlabel(DELTA_XLABEL, fontsize=DELTA_AXIS_FONT)
        ax.set_ylabel(DELTA_YLABEL, fontsize=DELTA_AXIS_FONT)
        ax.tick_params(labelsize=DELTA_TICK_FONT, length=2, pad=1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=DELTA_MAX_TICKS))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=DELTA_MAX_TICKS))

        # In-axes key. Plain text, not a legend, so constrained layout leaves
        # the axes at full width. It sits in the headroom reserved above.
        if show_key:
            key = [(f"{DELTA_YLABEL} > {DELTA_XLABEL}  n={n_up:,}",
                    AGED_COL if DELTA_SPLIT_COLOURS else DELTA_SIG_COL),
                   (f"{DELTA_XLABEL} > {DELTA_YLABEL}  n={n_dn:,}",
                    YOUNG_COL if DELTA_SPLIT_COLOURS else DELTA_SIG_COL)]
            if DELTA_KEY_SHOW_OTHER:
                key.append((f"other  n={n_ns:,}", "0.55"))
            y_cur = 0.995
            for txt, col in key:
                txt = "\n".join(textwrap.wrap(txt, DELTA_KEY_WRAP)) or txt
                ax.text(0.02, y_cur, txt, transform=ax.transAxes,
                        ha="left", va="top", fontsize=DELTA_KEY_FONT,
                        color=col, zorder=7)
                y_cur -= 0.075 * (txt.count("\n") + 1)
        despine(ax)

        # Keep the historical "_top20" stem when 20 genes really are labelled,
        # so existing references do not break; name it honestly otherwise.
        stem = make_plot_stem(
            7, group_label,
            (f"compare_age_groups_top{n_shown}" if with_labels
             else "compare_age_groups"),
            target_col)
        save_figure(fig, stem, out_dir)
        plt.close(fig)

    return d

# =============================================================================
# STEP 9: ONE GROUP (GLOBAL OR ONE COMPARTMENT)
# =============================================================================

def run_group_analysis(counts, meta, inv, n_points, gene_names, target_col,
                       out_dir, group_label, bin_mask=None,
                       point_mask=None, matrisome_df=None, mechanomics=None,
                       prerank_gene_sets=None, global_gene_mask=None):
    """
    Full point-level pipeline for one group. Returns a summary dict.

    bin_mask    which 4 um bins contribute expression (compartment bins only)
    point_mask  which points are included (compartment fraction >= threshold)
    """
    print("\n" + "=" * 70)
    print(f"GROUP: {group_label}   TARGET: {target_col}")
    print("=" * 70)
    out_dir.mkdir(parents=True, exist_ok=True)

    lognorm, raw_sum, pmeta = aggregate_expression_to_points(
        counts, meta, inv, n_points, target_col, bin_mask=bin_mask)

    keep = pmeta["n_bins_used"].values >= MIN_BINS_PER_POINT
    keep &= pmeta[target_col].notna().values
    if point_mask is not None:
        keep &= np.asarray(point_mask, dtype=bool)

    n_keep = int(keep.sum())
    print(f" Points retained: {n_keep:,} "
          f"(>= {MIN_BINS_PER_POINT} bins, valid {target_col}"
          + (", compartment fraction filter" if point_mask is not None else "")
          + ")")
    if n_keep < MIN_POINTS_PER_COMPARTMENT:
        print(f" Fewer than {MIN_POINTS_PER_COMPARTMENT} points; group skipped.")
        return None

    lognorm = lognorm.tocsr()[keep].tocsc()
    raw_sum = raw_sum.tocsr()[keep].tocsr()
    pmeta = pmeta.loc[keep].reset_index(drop=True)

    y = pmeta[target_col].values.astype(float)
    if LOG_TARGET and target_col in LOG_TARGETS:
        if np.nanmin(y) <= 0:
            print(f" [WARN] {int((y <= 0).sum())} non-positive {target_col} "
                  f"value(s); clipped before log10.")
        y = np.log10(np.maximum(y, 1e-12))

    z = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)
    n_out = int((np.abs(z) > 10).sum())
    if n_out:
        print(f" [WARN] {n_out} point(s) more than 10 SD from the mean in "
              f"{target_col}. Spearman is rank-based so the statistics are "
              f"unaffected, but the raw values need a QC pass.")

    log_depth = pmeta["log_depth"].values
    animals = pmeta["animal"].values if "animal" in pmeta.columns \
        else np.array(["one"] * n_keep)
    is_young = (pmeta["age_group"].values == "young")
    n_y, n_a = int(is_young.sum()), int((~is_young).sum())

    print(f" Age: young {n_y:,}, aged {n_a:,}   "
          f"Animals: {len(np.unique(animals))}   "
          f"median bins/point: {np.median(pmeta['n_bins_used'].values):.0f}")
    print(f" Depth per point: median {np.median(pmeta['depth'].values):,.0f} "
          f"counts")

    detect = np.asarray((raw_sum > 0).sum(axis=0)).ravel() / max(n_keep, 1)
    total = np.asarray(raw_sum.sum(axis=0)).ravel()
    gmask = (detect >= GENE_MIN_DETECT_FRAC) & (total >= GENE_MIN_TOTAL_COUNT)
    if global_gene_mask is not None and GENE_FILTER_SCOPE == "global":
        gmask = np.asarray(global_gene_mask, dtype=bool)
    n_genes = int(gmask.sum())
    print(f" Genes tested: {n_genes:,} of {len(gene_names):,} "
          f"(detected in >= {GENE_MIN_DETECT_FRAC:.0%} of points and "
          f">= {GENE_MIN_TOTAL_COUNT} counts; scope = {GENE_FILTER_SCOPE})")
    if n_genes == 0:
        print(" No genes pass the filter; group skipped.")
        return None

    Xg = lognorm[:, gmask].tocsc()
    gsel = np.asarray(gene_names)[gmask]

    print(f" Correlations use only points with per-gene expression >= "
          f"{EXPRESSION_SUBSET_MIN_COUNT} raw count(s)"
          f" (mode = {EXPRESSION_SUBSET_MODE}).")
    print(f" Covariates per gene: log10 depth"
          + (" + age group in pooled models" if AGE_AS_COVARIATE and n_y and n_a
             else "")
          + "; models are rebuilt on each gene's detected-point subset.")

    res = compute_point_correlations(Xg, raw_sum[:, gmask].tocsc(), gsel,
                                     y, log_depth, animals, is_young,
                                     detect[gmask], total[gmask])
    res = res.sort_values("p_global")
    res.to_csv(out_dir / f"{group_label}_{target_col}_pointlevel_correlations.csv",
               index=False)
    export_significant_only_rho_sheet(res, out_dir, target_col, group_label=group_label)

    print(f"\n Volcanoes ({len(VOLCANO_STATS)} statistics):")
    n_for_stat = {"global": n_keep, "within_animal": n_keep,
                  "young": n_y, "aged": n_a, "delta": n_keep}
    per_stat = {}
    for stat in VOLCANO_STATS:
        thr = RHO_DIFF_THRESH if stat == "delta" else RHO_THRESH
        sub = "depth-adjusted" if stat == "global" else ""
        if stat == "within_animal":
            sub = "depth + animal adjusted"
        elif stat == "delta":
            sub = "age-by-stiffness interaction"
        elif stat in ("young", "aged"):
            sub = f"{stat} points only, depth + animal adjusted"
        out = plot_volcano_correlations(
            res, stat, out_dir, target_col, group_label=group_label,
            mechanomics=mechanomics, rho_thresh=thr, fdr_thresh=FDR_THRESH,
            n_points=n_for_stat.get(stat, n_keep), subtitle=sub)
        per_stat[stat] = out

    main_stat = RANK_COL_FOR_PATHWAYS.replace("rho_", "")
    stats_to_run = list(DOWNSTREAM_STATS) if DOWNSTREAM_STATS else [main_stat]
    stats_to_run = [s for s in stats_to_run if s in per_stat]
    if main_stat not in stats_to_run:
        stats_to_run.insert(0, main_stat)

    # The summary at the end still reports the main statistic, so the returned
    # counts mean the same thing they always did.
    d_main, sig_pos_df, sig_neg_df, sig_combined, sig_genes = \
        per_stat.get(main_stat, (None, None, None, None, []))

    print(f"\n Downstream analyses for {len(stats_to_run)} statistic(s): "
          f"{stats_to_run}")
    print(f"   Each is a SEPARATE test. 'young' means this group's points "
          f"from young mice only, depth + animal adjusted within that stratum, "
          f"so {group_label}-within-young is not a subset of "
          f"{group_label}-overall.")

    genes_by_stat = {}
    for stat in stats_to_run:
        s_pack = per_stat.get(stat)
        if s_pack is None:
            print(f"\n [SKIP] {stat}: no volcano output")
            continue
        d_s, s_pos, s_neg, s_comb, s_genes = s_pack
        rho_col = f"rho_{stat}"
        # "global" keeps its historic filenames so nothing already curated is
        # renamed; every other statistic is tagged.
        tag = "" if (stat == main_stat and not STAT_SUFFIX_ON_GLOBAL) \
            else f"_{stat_disp(stat)}"
        print(f"\n{'=' * 60}")
        print(f" STATISTIC: {stat}   ({group_label}, {target_col})")
        print(f"{'=' * 60}")

        if s_comb is not None and len(s_comb):
            export_significant_gene_reference_table(
                Xg, gsel, pmeta, s_genes, out_dir, target_col,
                group_label=f"{group_label}{tag}")
            export_matrisome_hits(s_comb, out_dir, target_col,
                                  group_label=f"{group_label}{tag}",
                                  matrisome_df=matrisome_df)
            if MECH_TABLE is not None:
                ann = s_comb.merge(
                    MECH_TABLE.drop(columns=["in_expression_matrix"],
                                    errors="ignore"),
                    on="gene", how="left")
                ann.to_csv(
                    out_dir / f"{group_label}_{target_col}_{stat_disp(stat)}"
                              f"_significant_genes_annotated.csv", index=False)
                n_m = int(ann["source"].notna().sum())
                print(f" Annotated significant genes: {n_m} of {len(ann)} are "
                      f"in the Mechanomics set "
                      f"-> {group_label}_{target_col}_{stat_disp(stat)}"
                      f"_significant_genes_annotated.csv")

        matrisome_fisher_test(res, stat, matrisome_df, mechanomics,
                              out_dir, target_col, group_label=group_label)

        # GSEA runs on a fixed set of comparisons: all three statistics for the
        # whole tissue, global only for each compartment. Seven runs in total.
        gsea_stats = (GSEA_STATS_WHOLE if group_label == WHOLE_GROUP_LABEL
                      else GSEA_STATS_COMPARTMENT)
        if stat not in gsea_stats:
            print(f" [SKIP] GSEA for {stat}: {group_label} runs GSEA on "
                  f"{gsea_stats} only (see GSEA_STATS_WHOLE / "
                  f"GSEA_STATS_COMPARTMENT).")
        elif rho_col in res.columns:
            run_pathway_analysis(res, out_dir, target_col,
                                 group_label=group_label,
                                 mechanomics=mechanomics,
                                 prerank_gene_sets=prerank_gene_sets,
                                 rank_col=rho_col)
        else:
            print(f" [SKIP] GSEA for {stat}: no {rho_col} column")

        # ---- ORA on the significant gene lists ------------------------------
        # Independent of GSEA and gated separately, because the two answer
        # different questions and a comparison you want ORA for is not
        # necessarily one you want GSEA for.
        if RUN_ORA:
            is_whole = (group_label == WHOLE_GROUP_LABEL)
            ora_stats = ORA_STATS_WHOLE if is_whole else ORA_STATS_COMPARTMENT
            group_ok = is_whole or (group_label in ORA_COMPARTMENTS)
            if not group_ok:
                print(f" [SKIP] ORA for {stat}: {group_label} is not in "
                      f"ORA_COMPARTMENTS = {tuple(ORA_COMPARTMENTS)}.")
            elif stat not in ora_stats:
                print(f" [SKIP] ORA for {stat}: {group_label} runs ORA on "
                      f"{ora_stats} only (see ORA_STATS_WHOLE / "
                      f"ORA_STATS_COMPARTMENT).")
            else:
                run_ora_analysis(res, out_dir, target_col,
                                 group_label=group_label, stat=stat,
                                 gene_sets=prerank_gene_sets,
                                 mechanomics=mechanomics)

        # ---- scatter panels: significant Mechanomics hits only -------------
        # Top PANEL_TOP_N_PER_DIRECTION by rho in each direction, ranked by
        # effect size. If no Mechanomics gene is significant, nothing is drawn.
        tested = set(gsel)
        mech_set = set(mechanomics or [])
        panel_genes = []
        n_pos_avail = n_neg_avail = 0
        # PANEL_ALL_SIGNIFICANT: every gene past the gate, so the panels match
        # significant_only_r_values.csv row for row. Otherwise the historic
        # behaviour, top N per direction among Mechanomics members only.
        for src, asc in ((s_pos, False), (s_neg, True)):
            if src is None or not len(src):
                continue
            # Two INDEPENDENT gates. Membership is governed only by
            # PANEL_REQUIRE_MECHANOMICS, the cap only by
            # PANEL_ALL_SIGNIFICANT. They were entangled before, so turning
            # off "all significant" silently re-imposed the Mechanomics
            # filter and you would have got the top 50 Mechanomics genes
            # rather than the top 50 overall.
            d_ = src[src["gene"].isin(mech_set)] if PANEL_REQUIRE_MECHANOMICS \
                 else src
            if asc:
                n_neg_avail = len(d_)
            else:
                n_pos_avail = len(d_)
            g_ = d_.sort_values("rho", ascending=asc)["gene"]
            panel_genes += (g_.tolist() if PANEL_ALL_SIGNIFICANT
                            else g_.head(PANEL_TOP_N_PER_DIRECTION).tolist())
        panel_genes = [g for g in dict.fromkeys(panel_genes) if g in tested]
        genes_by_stat[stat] = list(panel_genes)

        if not panel_genes:
            # An empty panel set is almost always a THRESHOLD outcome, not a
            # failure, and the old one-line message could not tell the two
            # apart. This reports each gate in turn so the run log says which
            # one emptied the set, and quotes the standard error of a Spearman
            # rho at this n, since a stratum inside a compartment has roughly
            # half the points of the group and so an SE about 1.4x larger.
            n_stat = n_for_stat.get(stat, n_keep)
            se = (np.sqrt(1.06 / (n_stat - 3)) if n_stat > 3 else np.nan)
            n_tested_stat = (int(res[rho_col].notna().sum())
                             if rho_col in res.columns else 0)
            print(f"\n Correlation panels ({stat}): none drawn for "
                  f"{group_label}.")
            if n_tested_stat == 0:
                print(f"   The statistic itself was never computed here: no "
                      f"usable {rho_col}. Most likely this group has fewer "
                      f"than MIN_POINTS_PER_STRATUM = "
                      f"{MIN_POINTS_PER_STRATUM} points in that age group, so "
                      f"no model was built. THIS is a real gap, not a "
                      f"threshold.")
            else:
                n_sig = (int((res[f"fdr_{stat}"] < FDR_THRESH).sum())
                         if f"fdr_{stat}" in res.columns else -1)
                n_eff = ((len(s_pos) if s_pos is not None else 0)
                         + (len(s_neg) if s_neg is not None else 0))
                print(f"   n = {n_stat:,} points, SE(rho) ~ {se:.3f}, so "
                      f"|rho| >= {RHO_THRESH} is about "
                      f"{RHO_THRESH / se:.1f} SE.")
                print(f"   {n_tested_stat:,} genes tested -> "
                      f"{n_sig:,} pass FDR < {FDR_THRESH} -> "
                      f"{n_eff:,} also pass |rho| >= {RHO_THRESH}"
                      + (" -> 0 of those are in the Mechanomics set."
                         if PANEL_REQUIRE_MECHANOMICS else "."))
                print(f"   Nothing is broken; the gate is simply not met. "
                      f"Widening it would manufacture hits, so it is left "
                      f"alone unless you change FDR_THRESH or RHO_THRESH "
                      f"deliberately.")
        else:
            how = ("all significant" if PANEL_ALL_SIGNIFICANT
                   else f"top {PANEL_TOP_N_PER_DIRECTION} per direction")
            src = " Mechanomics" if PANEL_REQUIRE_MECHANOMICS else ""
            print(f"\n Correlation panels ({stat}): {len(panel_genes)} gene(s) "
                  f"({how} of {n_pos_avail} positive / {n_neg_avail} negative"
                  f"{src} hits)")

    # ---- draw, once the per-statistic lists are all known -----------------
    # PANEL_GENES_UNION_ACROSS_STATS makes the three panels comparable: a gene
    # selected by ANY statistic gets a panel for EVERY statistic. Without it
    # the lists are built independently and are largely disjoint, so a gene can
    # have an agedcorr panel and no globalcorr one.
    if genes_by_stat:
        union = list(dict.fromkeys(
            [g for lst in genes_by_stat.values() for g in lst]))
        for stat, own in genes_by_stat.items():
            # A statistic with no hits of its own gets no panels, even though
            # the union is not empty. Otherwise CL agedcorr, which passes
            # nothing, would draw the whole union.
            if PANEL_SKIP_STAT_WITH_NO_HITS and not own:
                print(f"\n Drawing 0 {stat} panel(s) for {group_label}: this "
                      f"statistic has no gene of its own past FDR < "
                      f"{FDR_THRESH} and |rho| >= {RHO_THRESH}, so nothing is "
                      f"carried in from the other statistics.")
                continue
            plot_list = union if PANEL_GENES_UNION_ACROSS_STATS else own
            if not plot_list:
                continue
            tag2 = "" if (stat == main_stat and not STAT_SUFFIX_ON_GLOBAL) \
                else f"_{stat_disp(stat)}"
            extra = len(plot_list) - len(own)
            print(f"\n Drawing {len(plot_list):,} {stat} panel(s)"
                  + (f" ({extra:,} carried in from the other statistics so the "
                     f"panels are comparable)" if extra > 0 else ""))
            if len(plot_list) > PANEL_CONFIRM_ABOVE:
                print(f"   [STOP] {len(plot_list):,} panels exceeds "
                      f"PANEL_CONFIRM_ABOVE = {PANEL_CONFIRM_ABOVE:,}. That is "
                      f"{2 * len(plot_list):,} files for this statistic alone "
                      f"(PNG + SVG). Nothing was drawn. Raise "
                      f"PANEL_CONFIRM_ABOVE once you are happy to wait, or set "
                      f"PANEL_ALL_SIGNIFICANT = False.")
                continue
            plot_gene_correlations(Xg, raw_sum[:, gmask].tocsc(), gsel,
                                   pmeta, res, out_dir,
                                   target_col, genes=plot_list,
                                   group_label=group_label, stat=stat,
                                   stem_tag=tag2)

    compare_aged_young(res, out_dir, target_col, group_label=group_label)

    return {
        "group": group_label,
        "n_points": n_keep,
        "n_points_young": n_y,
        "n_points_aged": n_a,
        "n_animals": int(len(np.unique(animals))),
        "n_genes_tested": n_genes,
        "n_sig_total": len(sig_genes) if sig_genes else 0,
        "n_sig_pos": len(sig_pos_df) if sig_pos_df is not None else 0,
        "n_sig_neg": len(sig_neg_df) if sig_neg_df is not None else 0,
        "max_abs_rho": float(res[RANK_COL_FOR_PATHWAYS].abs().max())
        if RANK_COL_FOR_PATHWAYS in res.columns else np.nan,
    }

# =============================================================================
# STEP 10: ONE TARGET, GLOBAL PLUS EVERY COMPARTMENT
# =============================================================================

def run_single_target_analysis(counts, gene_names, expr_dir, metadata_tsv,
                               out_root, target_col, matrisome_df=None,
                               mechanomics=None, prerank_gene_sets=None):
    print("\n" + "#" * 70)
    print(f"RUNNING ANALYSIS FOR TARGET: {target_col}")
    print("#" * 70)

    target_out = out_root / target_col
    target_out.mkdir(parents=True, exist_ok=True)

    meta = load_matched_metadata(expr_dir, metadata_tsv, target_col)
    if "expr_row" not in meta.columns:
        raise ValueError("Metadata is missing 'expr_row'.")

    expr_row_idx = meta["expr_row"].to_numpy(dtype=int)
    if expr_row_idx.max() >= counts.shape[0]:
        raise ValueError(
            f"expr_row index exceeds matrix rows: max={expr_row_idx.max()}, "
            f"rows={counts.shape[0]}")
    counts_sub = counts[expr_row_idx, :]
    meta = meta.reset_index(drop=True)

    print(f"\nAligned matrix: {counts_sub.shape[0]:,} 4 um bins x "
          f"{counts_sub.shape[1]:,} genes")

    _, inv, _ = build_point_index(meta)
    n_points = int(inv.max()) + 1

    _bin_tab, frac_tab = compartment_fractions(meta, inv, n_points)
    print("\n Compartment composition of the 50 um points "
          f"(threshold >= {COMPARTMENT_MIN_FRAC:.0%} of bins):")
    for c in frac_tab.columns:
        n_pass = int((frac_tab[c] >= COMPARTMENT_MIN_FRAC).sum())
        print(f"   {str(c):<10s} mean fraction {frac_tab[c].mean():.3f}, "
              f"{n_pass:,} point(s) pass")
    n_multi = int((frac_tab >= COMPARTMENT_MIN_FRAC).sum(axis=1).gt(1).sum())
    if n_multi:
        print(f"   {n_multi:,} point(s) reach the threshold in more than one "
              f"compartment and will appear in both analyses, as intended.")

    # ---- global gene mask, used only when GENE_FILTER_SCOPE == "global" ----
    _ln_all, raw_all, pm_all = aggregate_expression_to_points(
        counts_sub, meta, inv, n_points, target_col, bin_mask=None)
    ok_all = (pm_all["n_bins_used"].values >= MIN_BINS_PER_POINT) & \
             pm_all[target_col].notna().values
    raw_ok = raw_all[ok_all]
    det_all = np.asarray((raw_ok > 0).sum(axis=0)).ravel() / max(int(ok_all.sum()), 1)
    tot_all = np.asarray(raw_ok.sum(axis=0)).ravel()
    global_gene_mask = (det_all >= GENE_MIN_DETECT_FRAC) & \
                       (tot_all >= GENE_MIN_TOTAL_COUNT)
    print(f"\n Genes passing the filter on ALL points: "
          f"{int(global_gene_mask.sum()):,} of {len(gene_names):,}")
    del _ln_all, raw_all, raw_ok, pm_all

    summaries = []

    # ---------------------------------------------------------------- global
    s = run_group_analysis(
        counts_sub, meta, inv, n_points, gene_names, target_col,
        out_dir=target_out / f"{WHOLE_GROUP_LABEL}_{target_col}",
        group_label=WHOLE_GROUP_LABEL,
        bin_mask=None, point_mask=None, matrisome_df=matrisome_df,
        mechanomics=mechanomics, prerank_gene_sets=prerank_gene_sets,
        global_gene_mask=global_gene_mask)
    if s:
        summaries.append(dict(target=target_col, **s))

    # --------------------------------------------------------- compartments
    comp_bins = meta["compartment_short"].astype(str).values
    for comp in COMPARTMENTS_TO_RUN:
        if comp not in frac_tab.columns:
            print(f"\n Compartment {comp}: not present in the metadata; skipped.")
            continue
        pmask = (frac_tab[comp].values >= COMPARTMENT_MIN_FRAC)
        bmask = (comp_bins == comp)
        n_pass = int(pmask.sum())
        if n_pass < MIN_POINTS_PER_COMPARTMENT:
            print(f"\n Compartment {comp}: only {n_pass} point(s) at "
                  f">= {COMPARTMENT_MIN_FRAC:.0%}; skipped "
                  f"(minimum {MIN_POINTS_PER_COMPARTMENT}).")
            continue
        s = run_group_analysis(
            counts_sub, meta, inv, n_points, gene_names, target_col,
            out_dir=target_out / f"{comp}_{target_col}", group_label=comp,
            bin_mask=bmask, point_mask=pmask, matrisome_df=matrisome_df,
            mechanomics=mechanomics, prerank_gene_sets=prerank_gene_sets,
            global_gene_mask=global_gene_mask)
        if s:
            summaries.append(dict(target=target_col, **s))

    return summaries

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
            print("Point-level stiffness-expression correlation run")
            print(f"Date/time: {datetime.now().isoformat(timespec='seconds')}")
            print(f"Metadata TSV: {METADATA_TSV}")
            print(f"Expression directory: {EXPR_DIR}")
            print(f"Output directory: {OUTPUT_DIR}")
            print(f"Primary targets: {PRIMARY_TARGETS}")
            print(f"AGE_AS_COVARIATE: {AGE_AS_COVARIATE}")
            print(f"Unit of observation: 50 um nanoindentation point")
            print(f"Point key: {' + '.join(POINT_KEY_COLS)}")
            print(f"Compartment threshold: >= {COMPARTMENT_MIN_FRAC:.0%} of "
                  f"a point's 4 um bins")
            print(f"Gene filter scope: {GENE_FILTER_SCOPE}")
            print(f"FDR < {FDR_THRESH}, |rho| >= {RHO_THRESH}")
            print(f"Pathways ranked by: {RANK_COL_FOR_PATHWAYS}")
            print(f"RANDOM_STATE: {RANDOM_STATE}")
            print(f"Run log: {log_path}")
            print("=" * 70)

            X, barcodes, gene_names, is_counts = load_expression_counts(EXPR_DIR)

            if not is_counts:
                mm = pd.read_csv(EXPR_DIR / EXPR_MATCHED_METADATA_FILE,
                                 sep="\t", low_memory=False)
                if len(mm) != X.shape[0]:
                    raise ValueError(
                        f"Matched metadata rows ({len(mm):,}) do not match "
                        f"matrix rows ({X.shape[0]:,}); counts cannot be "
                        f"reconstructed.")
                if DEPTH_COL not in mm.columns:
                    raise ValueError(
                        f"'{DEPTH_COL}' not in the matched metadata export, so "
                        f"LogNormalize cannot be inverted. Either add it or "
                        f"export raw counts as {EXPR_COUNTS_FILE}.")
                X = reconstruct_counts(
                    X, pd.to_numeric(mm[DEPTH_COL], errors="coerce").values)
                del mm

            matrisome_df = None
            if RUN_MATRISOME:
                try:
                    matrisome_df = load_matrisome_table(MATRISOME_CSV)
                    print(f"\nLoaded matrisome table: {MATRISOME_CSV} "
                          f"({len(matrisome_df):,} rows, "
                          f"{matrisome_df[MATRISOME_COLS['gene']].nunique():,} symbols)")
                except Exception as e:
                    print(f"WARNING: failed to load matrisome table: {e}")

            goi_df = load_goi_table(GOI_CSV)
            lr_df = load_lr_pairs(LR_PAIRS_CSV)
            mech_table, mechanomics = build_mechanomics_table(
                matrisome_df, lr_df, goi_df, universe=gene_names)
            mech_table.to_csv(OUTPUT_DIR / "mechanomics_gene_set.csv",
                              index=False)
            globals()["MECH_TABLE"] = mech_table
            print(" Written: mechanomics_gene_set.csv")

            goi = sorted(set(goi_df["gene"])) if goi_df is not None else []
            missing_goi = [g for g in goi if g not in set(gene_names)]
            if missing_goi:
                print(f" NOTE: {len(missing_goi)} of {len(goi)} gene(s) of "
                      f"interest are not in the expression matrix: "
                      f"{missing_goi[:30]}"
                      f"{' ...' if len(missing_goi) > 30 else ''}")
                pd.DataFrame({"gene": missing_goi}).to_csv(
                    OUTPUT_DIR / "goi_not_in_expression_matrix.csv", index=False)

            missing_panel = [g for g in MANUAL_GENE_PLOT_LIST
                             if g not in set(gene_names)]
            if missing_panel:
                print(f" NOTE: {len(missing_panel)} scatter-panel gene(s) are "
                      f"not in the expression matrix: {missing_panel}")

            # Loaded once per run and shared by GSEA and ORA. Previously this
            # was gated on RUN_GSEA alone, so with GSEA off ORA would have
            # re-downloaded the libraries once per group.
            prerank_gene_sets = get_prerank_gene_sets(mechanomics) \
                if ((RUN_GSEA or RUN_ORA) and GSEAPY_AVAILABLE) else None

            # The volcano legend, once, as its own file. It does not fit inside
            # a 35 mm panel (it measures 42 mm wide at 6 pt).
            if not VOLCANO_LEGEND:
                save_volcano_legend(OUTPUT_DIR)

            all_summaries = []
            for target_col in PRIMARY_TARGETS:
                try:
                    all_summaries += run_single_target_analysis(
                        counts=X,
                        gene_names=gene_names,
                        expr_dir=EXPR_DIR,
                        metadata_tsv=METADATA_TSV,
                        out_root=OUTPUT_DIR,
                        target_col=target_col,
                        matrisome_df=matrisome_df,
                        mechanomics=mechanomics,
                        prerank_gene_sets=prerank_gene_sets,
                    )
                except Exception as e:
                    import traceback
                    print(f"\nERROR while processing {target_col}: {e}")
                    traceback.print_exc()

            print("\n" + "=" * 70)
            print("ANALYSIS SUMMARY")
            print("=" * 70)
            if all_summaries:
                sdf = pd.DataFrame(all_summaries)
                sdf.to_csv(OUTPUT_DIR / "analysis_summary_by_group.csv",
                           index=False)
                print(sdf.to_string(index=False))
                print("\nSaved: analysis_summary_by_group.csv")
            else:
                print("No groups completed successfully.")

            print(f"\nAll outputs saved to: {OUTPUT_DIR}")
            if not AGE_AS_COVARIATE:
                print("\nNEXT: set AGE_AS_COVARIATE = True at the top of this "
                      "script and run again.")
                print("That writes to the sibling folder ..._ageAdj and leaves "
                      "this one untouched.")
            print("=" * 70)
            print(f"Run log saved to: {log_path}")

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()