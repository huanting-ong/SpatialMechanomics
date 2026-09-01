"""
plot_segmentation.py
====================
Spatial square-dot plots of ROI-Segmentation metadata.
All per-tissue sub-plots share the same fixed canvas and axis limits so
tissues appear at consistent proportional size when trimmed.

Colour scheme
-------------
F  (Follicle)        #af5420  — numbered sub-features, dark → light shading
CL (Corpus luteum)   #8e6bd0  — numbered sub-features, dark → light shading
S  (Stroma + OSE)    #54a397  — single colour
V  (Vascular/Immune) #b44769  — single colour
Unassigned           omitted (grey shadow only)

THE GREY SHADOW SHOWS ONLY THE SLICE THAT CARRIES THE ROI
---------------------------------------------------------
An OvaryID can cover two serial sections on the same capture area. Only one of
them is usually annotated, so drawing every bin of that OvaryID put a second,
empty tissue outline next to the real plot and pushed the annotated slice into
a corner. Per-tissue plots are now restricted to the Tissue-slices value(s)
that actually contain assigned ROI-Segmentation bins, which also tightens the
axis limits onto the annotated section. Set RESTRICT_TO_ROI_SLICES = False to
get the old behaviour back.

Outputs per tissue (OvaryID: Y1, Y2, A3, …):
  <ID>-F               follicles, shaded; grey shadow of the ROI slice behind
  <ID>-CL              CLs, shaded; shadow behind
  <ID>-S               stroma; shadow behind
  <ID>-V               vascular/immune; shadow behind
  <ID>-ROI-Segmentation  all four compartments; shadow behind
       -> PNG 600 dpi · TIFF 600 dpi · SVG

Illustrator layers, per tissue (no grey shadow, no axes, title only):
  <ID>_Fonly  <ID>_CLonly  <ID>_Sonly  <ID>_Vonly
  <ID>-F1  <ID>-F2 … <ID>-CL1  <ID>-CL2 …   one file per numbered feature
       -> PNG 600 dpi · SVG

  These share the tissue's axis limits and canvas and are saved WITHOUT tight
  bounding boxes, so every layer of a given tissue registers pixel-for-pixel
  when stacked in Illustrator. Backgrounds are transparent by default
  (FEATURE_ONLY_TRANSPARENT).

Per tissue-slice (Tissue-slices column: Y1-1, Y1-2, A3-2, …):
  <Slice>-Compartments   all bins coloured by Compartments column

Global:
  All_CaptureAreas_Compartments   all 4 CAs by Compartments column
"""

import re
import sys
import colorsys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from scipy import ndimage
import sys as _sys
import os
import io
import time
import shutil
import tempfile
import traceback
import multiprocessing as mp
from contextlib import redirect_stdout
from datetime import datetime


class Tee:
    """Mirror stdout to the console and to the run log at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for st in self.streams:
            st.write(data)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()

# =============================================================================
# CONFIGURATION
# =============================================================================

# ---- Option A: raw SpatialX metadata (default) ----
METADATA_TSV = Path(
    r"Z:\Visium_Data\001_SpatialX\20260803_FinalAnnotatedMetadata"
    r"\HTO_Combined_4um_Images\metadata"
    r"\HTO_Combined_4um_Images_annotated_data.tsv"
)

# ---- Option B: step9 expression-merged metadata ----
# Uncomment below and comment out Option A.
# Column names are identical in the step9 output.
# METADATA_TSV = Path(
#     r"Z:\Visium_Data\001_SpatialX\20260803_FinalAnnotatedMetadata"
#     r"\expression_export\step9_expression_metadata_matched.tsv"
# )

OUTPUT_DIR = METADATA_TSV.parent.parent.parent / "Segmentation"

# ---- Output organisation --------------------------------------------------
# True sorts every image into Young/ or Aged/ and then into one of four
# subfolders. False writes everything flat into OUTPUT_DIR, which is what the
# previous version did.
#
#   Segmentation/
#     colourbar_*.png|svg                     colour bars, shared by both ages
#     All_CaptureAreas_Compartments.*         spans both ages, so it stays here
#     run_log_*.txt
#     Young/
#       01_WholeTissue_ROI_CaptureArea/   <ID>-<comp>, <ID>-ROI-Segmentation,
#                                         <slice>-Compartments
#       02_STbins_and_Halos/              <ID>_<comp>only, <ID>-<comp><n>,
#                                         <ID>-<comp><n>_halo
#       03_Stiffness/                     every *_Elasticity_l_Pa layer
#       04_Genes/                         every *_<gene> layer
#     Aged/
#       ... the same four
#
# Age comes from the leading letter of OvaryID / Tissue-slices: Y -> Young,
# A -> Aged. Anything else lands in UNGROUPED_DIR rather than being guessed at,
# so a mislabelled tissue is visible instead of silently filed under one age.
ORGANISE_OUTPUT = True

DIR_YOUNG     = "Young"
DIR_AGED      = "Aged"
DIR_UNGROUPED = "Ungrouped"

# True splits every layer kind into a whole-compartment folder and a numbered
# F / CL feature folder, because the two are used at different points in a
# figure and there are far more of the second. False gives the previous four
# folders, with both levels mixed together.
SPLIT_COMPARTMENT_AND_FEATURES = True

SUBDIR_OVERVIEW = "01_WholeTissue_ROI_CaptureArea"

# split layout
SUBDIR_COMP       = "02_Compartment_layers"
SUBDIR_FEAT       = "03_Individual_features"
SUBDIR_STIFF_COMP = "04_Stiffness_compartments"
SUBDIR_STIFF_FEAT = "05_Stiffness_features"
SUBDIR_GENE_COMP  = "06_Genes_compartments"
SUBDIR_GENE_FEAT  = "07_Genes_features"

# unsplit layout, used when SPLIT_COMPARTMENT_AND_FEATURES = False
SUBDIR_FEATURES  = "02_STbins_and_Halos"
SUBDIR_STIFFNESS = "03_Stiffness"
SUBDIR_GENES     = "04_Genes"

# ---- Gene layers get their own tree ---------------------------------------
# Every spatial expression plot lives under GENES_ROOT_DIR, so deleting that
# one folder before a run clears all of them and touches nothing else.
#
# GENE_GROUP_BY decides how that tree is subdivided. Every filename carries
# BOTH the tissue and the gene either way, so no grouping loses information;
# it only changes what sits next to what.
#
#   "tissue"  one folder per tissue, holding every gene for that tissue. This
#             is the QC view: open Y1 and see everything plotted on that
#             section at once, so a registration or masking problem that
#             affects the whole section is obvious.
#
#               Segmentation/Genes/Y1/
#                   Y1_Sonly_Col1a1_greyROI.svg + .png
#                   Y1_Fonly_Col1a1_greyROI.svg + .png
#                   Y1_Vonly_Fbln5_greyROI.svg  + .png
#
#   "gene"    one folder per gene, holding both ages and every compartment that
#             gene is a DEG in. The figure-assembly view: everything you need
#             for one gene's panel in one place.
#
#               Segmentation/Genes/Col1a1/
#                   Y1_Sonly_Col1a1_greyROI.svg   A3_Sonly_Col1a1_greyROI.svg
#                   Y1_Fonly_Col1a1_greyROI.svg   A3_Fonly_Col1a1_greyROI.svg
#
#   "flat"    no subdivision, everything in GENES_ROOT_DIR.
#
# Switching is one line, but the old tree is NOT cleaned up automatically, so
# delete GENES_ROOT_DIR when you change this or you will have both layouts side
# by side. The run warns about exactly that.
GENE_GROUP_BY = "tissue"
GENES_ROOT_DIR = "Genes"
WARN_ON_EXISTING_GENE_FILES = True

# ---- Hand-picked genes ----------------------------------------------------
# A separate, independent set of spatial maps for genes you name yourself,
# written to their own folder and NOT subject to the DEG machinery. Empty list
# turns the whole thing off.
#
# These differ from the DEG-driven gene layers in three ways:
#   - they go to HANDPICKED_DIR, not into the Genes/ tree
#   - they are drawn in EVERY compartment present, ignoring GENE_COMPARTMENTS,
#     because the point of hand-picking is usually to look somewhere the DEG
#     test did not send you
#   - they can be restricted to particular tissues
#
# A gene may appear here AND in the DEG-driven list; the two sets of files are
# in different folders and are computed independently, so neither overwrites
# the other.
# Mouse symbols are Title case (Col1a1), not human all-caps (COL1A1). Every
# symbol below that carries a "verified" note was checked against the
# in_expression_matrix column of your own mechanomics_gene_set.csv, so it is
# confirmed present in YOUR matrix, not merely a real gene somewhere.
#
# Four corrections to the list as given:
#
#  1. Col1a3 and Col1a4 DO NOT EXIST. Type I collagen is a heterotrimer of two
#     alpha-1(I) chains and one alpha-2(I) chain, so the genes are Col1a1 and
#     Col1a2 and there is no third or fourth chain.
#
#  2. Col3a2, Col3a3 and Col3a4 DO NOT EXIST. Type III collagen is a homotrimer
#     of three identical alpha-1(III) chains, so Col3a1 is the whole story.
#
#  3. TAZ is the one to be careful about. The Hippo co-activator TAZ is the
#     gene Wwtr1. There is ALSO a mouse gene literally named Taz, tafazzin,
#     which is a mitochondrial phospholipid transacylase and the Barth syndrome
#     gene. It has nothing to do with mechanotransduction. Asking for "Taz"
#     silently returns that one. Wwtr1 is used below.
#     YAP is likewise Yap1, not Yap.
#
#  4. "Laminin" is a family, not a gene: laminins are alpha/beta/gamma
#     heterotrimers. All eleven mouse chain genes are in your matrix and all
#     eleven are listed. Trim to the ones you care about if 11 panels is too
#     many; Lama4, Lamb1 and Lamc1 are the usual basement-membrane workhorses.
HANDPICKED_GENES = [
    # Collagen I  (verified present)
    "Col1a1", "Col1a2",
    # Collagen III  (verified present)
    "Col3a1",
    # Collagen IV  (verified present)
    "Col4a1", "Col4a2", "Col4a3", "Col4a4",
    # Col4a5 and Col4a6 also exist and are also in your matrix; uncomment for
    # the complete type IV set.
    # "Col4a5", "Col4a6",
    # Collagen VI  (verified present, including Col6a4, which is a pseudogene
    # in human but a real gene in mouse)
    "Col6a1", "Col6a2", "Col6a3", "Col6a4",
    # Col6a5 and Col6a6 also exist and are also in your matrix.
    # "Col6a5", "Col6a6",
    # Fibronectin  (verified present)
    "Fn1",
    # Integrins alpha-5, alpha-6, beta-1  (verified present)
    "Itga5", "Itga6", "Itgb1",
    # Hippo pathway. Yap1 and Wwtr1 verified present. Tead1 is a real mouse
    # gene but is not on your matrisome/GOI lists, so I could not confirm it
    # against your matrix from here; the run will say if it is absent.
    "Yap1", "Wwtr1", "Tead1",
    # Connexin 43, inhibin alpha, smooth muscle actin  (verified present)
    "Gja1", "Inha", "Acta2",
    # Laminin chains  (all verified present)
    "Lama1", "Lama2", "Lama3", "Lama4", "Lama5",
    "Lamb1", "Lamb2", "Lamb3",
    "Lamc1", "Lamc2", "Lamc3",
]
HANDPICKED_TISSUES = ["Y4"]        # [] or None = every tissue
HANDPICKED_COMPARTMENTS = None     # None = every compartment present
HANDPICKED_DIR = "handpickedgenes"

# Look each hand-picked gene up in the compartment DEG tables and report
# whether it would have made the DEG list, and if not, what its log2FC and
# padj actually are. Written to the run log and to a CSV.
HANDPICKED_REPORT_DEG_STATUS = True

# (kind, level) -> folder.  level is "compartment" for a whole-compartment
# export and "feature" for a numbered F / CL one.
KIND_TO_SUBDIR_SPLIT = {
    ("overview",  None):          SUBDIR_OVERVIEW,
    ("features",  "compartment"): SUBDIR_COMP,
    ("features",  "feature"):     SUBDIR_FEAT,
    ("stiffness", "compartment"): SUBDIR_STIFF_COMP,
    ("stiffness", "feature"):     SUBDIR_STIFF_FEAT,
    ("gene",      "compartment"): SUBDIR_GENE_COMP,
    ("gene",      "feature"):     SUBDIR_GENE_FEAT,
}
KIND_TO_SUBDIR = {
    "overview":  SUBDIR_OVERVIEW,
    "features":  SUBDIR_FEATURES,
    "stiffness": SUBDIR_STIFFNESS,
    "gene":      SUBDIR_GENES,
}

# Column names
COL_X           = "x"
COL_Y           = "y"
COL_SEG         = "ROI-Segmentation"   # "Y1-F-3", "Y1-S", "Unassigned" …
COL_COMPARTMENT = "Compartments"       # F / CL / S / V (for slice/all-CA maps)
COL_OVARY_ID    = "OvaryID"            # Y1, A3 … (tissue grouper)
COL_TISSUE_SLICE= "Tissue-slices"      # Y1-1, Y1-2, A3-2 … (slice grouper)

# Compartment base colours — applied exactly as provided
BASE_COLOURS = {
    "CL": "#8e6bd0",
    "F":  "#af5420",
    "S":  "#54a397",
    "V":  "#b44769",
}
SHADOW_COLOUR = "#d8d8d8"   # light grey for full-tissue background shadow

# Fixed canvas for all per-tissue plots (inches)
CANVAS_W   = 12.0    # plot area + legend strip
CANVAS_H   = 9.0
LEGEND_W   = 3.0     # reserved for the legend on the right

# ---- Spatial scale --------------------------------------------------------
# "global": every tissue is drawn over the SAME axis span in micrometres,
# centred on its own middle. One micrometre is then the same number of pixels
# in every image, so a follicle in Y1 and a follicle in A3 can be compared by
# eye and measured off the figure. The span is the largest tissue's, so nothing
# is cropped.
# "tissue": each tissue gets its own limits, which fills the canvas better but
# silently magnifies the small tissues. That was the old behaviour.
EXTENT_SCOPE = "global"
EXTENT_PAD   = 0.04          # fraction of the span added as margin

# ---- Feature shade consistency -------------------------------------------
# "global": CL3 is the same purple in every tissue, because the lightness ramp
# is built over the highest feature NUMBER seen anywhere and then indexed by
# number. Required if the panels sit side by side in a figure.
# "tissue": ramp built from the count of features in that tissue, so CL3 in a
# 7-CL ovary is a different purple from CL3 in a 3-CL ovary.
SHADE_SCOPE = "global"

# Marker fill factor: 4.5 µm squares on 4 µm bin centres → slight overlap,
# ensuring no visible gaps between adjacent bins.
BIN_FILL_UM = 4.5
BIN_OVERLAP = 1.12
BIN_PITCH_FALLBACK_UM = 4.0
BIN_PITCH_SAMPLE = 20000

SKIP = {"Unassigned", "unassigned", "N/A", "nan", "None", "none", ""}

PNG_DPI  = 600
TIFF_DPI = 600

# ---- Which file formats to write ------------------------------------------
# SVG only. The dots inside it are rasterised (see RASTERIZE_DOTS below), so
# the file stays small while the text and the layer structure stay editable.
# Put "png" and/or "tiff" back in the tuple if you need them for a quick look;
# order does not matter. The colour bars follow the same setting.
EXPORT_FORMATS = ("svg",)

# Per-kind override. A kind listed here uses its own formats instead of
# EXPORT_FORMATS. The gene layers get a 600 dpi PNG alongside the SVG because
# they are the ones you look at directly rather than assembling in Illustrator,
# and a rasterised-dot SVG is awkward to skim at a glance.
EXPORT_FORMATS_BY_KIND = {
    "gene": ("svg", "png"),
    "handpicked": ("svg", "png"),
}

# ---- Parallel rendering ---------------------------------------------------
# Nearly all the wall-clock here is matplotlib laying out and writing figures,
# one process at a time. That work is CPU-bound Python, so threads would be
# blocked by the GIL and give almost nothing; separate processes are needed.
#
# The unit of parallelism is ONE call to a plotting function, e.g. "Y1's CL
# compartment layer" or "A3's Fbln5 value layers". That is finer than one job
# per tissue: with 8 tissues, per-tissue jobs would cap the speedup at 8x no
# matter how many cores you have. Per plotting call there are typically several
# hundred jobs, so 60 workers stay fed.
#
# Set to 1 (or 0) to go back to serial, which is also the automatic fallback if
# the pool cannot start. Set to None for os.cpu_count(). The value is capped at
# os.cpu_count() and at the number of jobs, and the actual figure used is
# printed, so asking for 60 on a 16-core machine is not silently ignored.
#
# Honest caveat: the outputs are written to a network drive (Z:). If that share
# becomes the bottleneck, adding workers past a point buys nothing, because the
# job stops being CPU-bound and becomes I/O-bound. The run prints throughput at
# the end so you can see which regime you are in.
N_WORKERS = 60
WORKER_MEMORY_FRACTION = 0.60
PARALLEL_MIN_JOBS = 4          # below this, serial is not worth the setup

# ---- Where the workers' scratch frames live -------------------------------
# The parallel path writes each tissue's DataFrame to a scratch file once, and
# the workers load it from there instead of it being pickled into every job.
# Those files are pure scratch and are deleted at the end of the run. You may
# have seen one survive as plotseg_frames_<random> next to your outputs.
#
# None = the machine's own temp directory, which is the right default for two
# reasons: it is on a LOCAL disk, so 60 workers are not all reading scratch
# over the network on every job, and it keeps scratch out of the folder you
# actually want to look at. Set a path here to override, e.g. a fast local
# scratch disk, or OUTPUT_DIR to restore the old behaviour.
FRAME_SCRATCH_DIR = None
# Sweep up plotseg_frames_* left behind by an earlier run that was interrupted.
CLEAN_STALE_SCRATCH = True

# ---- Axes on the overview plots -------------------------------------------
# False switches the axes off on the whole-tissue, ROI and capture-area maps,
# so they match the Illustrator layers. The title and the legend are separate
# artists and still draw. The scale bar is then gone from the figure, so if you
# need a micrometre reference on these, note that EXTENT_SCOPE = "global" means
# every tissue is drawn over the same span, printed in the run log.
OVERVIEW_SHOW_AXES = False

# ---- Background -----------------------------------------------------------
# True gives every export an alpha channel with nothing behind the ink, so PNG,
# TIFF and SVG all drop onto whatever is underneath them in Illustrator.
#
# Two things this does NOT make transparent, deliberately:
#   - the legend box on the overview plots, which keeps its own opaque patch so
#     the labels stay readable over tissue. Set LEGEND_FRAME = False to drop it.
#   - the colour bars, which are read as standalone figures. Set
#     COLOURBAR_TRANSPARENT = True if you want those transparent too.
TRANSPARENT_BACKGROUND = True
LEGEND_FRAME           = True
COLOURBAR_TRANSPARENT  = False

# ---- Rasterised dots ------------------------------------------------------
# Each export draws one square marker per 4 um bin, tens of thousands of them.
# As vectors that is tens of thousands of separate paths in the SVG, which is
# what makes the files large and Illustrator slow to open them.
#
# RASTERIZE_DOTS embeds the dots as a single image inside the SVG instead. The
# title, the axes and the legend stay vector and stay editable. The PNG and
# TIFF are unaffected, they were always raster.
#
# The trade: the dots are no longer individually selectable in Illustrator and
# cannot be recoloured there. If you need to recolour a layer by hand, set this
# to False for that run.
RASTERIZE_DOTS  = True
RASTERIZE_HALOS = False    # halos are one smooth polygon each, already small
RASTER_DPI      = 600      # resolution of the embedded image inside the SVG

# ---- Grey ROI background on the layer exports -----------------------------
# A grey silhouette behind a compartment layer, so the layer can be read on its
# own without the rest of the ovary having to be imported alongside it.
#
# GREY_BG_SCOPE decides what the silhouette is:
#   "roi"    every bin with an assigned ROI-Segmentation label, i.e. the four
#            compartments merged. The annotated region only.
#   "tissue" every bin in the ROI-bearing slice, annotated or not, which is
#            the whole section outline.
# I read "the entire ROI" as the first. Change the one word if you meant the
# whole section.
GREY_BG_SCOPE = "roi"

# How the grey is delivered for the whole-compartment layers:
#
#   "two_files"  Two files per layer. <ID>_Fonly is the dots alone on a
#                transparent canvas; <ID>_Fonly_greyROI is the same dots with
#                the grey merged into ONE rasterised image. Both are drawn at
#                identical limits so they register.
#   "layered"    One file, grey in its own Axes so it stays independently
#                movable in Illustrator. Two <image> elements, axes_1 (grey)
#                and axes_2 (the layer).
#   "none"       No grey anywhere.
#
# Measured SVG size, 12,000-bin ROI, 600 dpi, one compartment layer:
#     layered (2 images in 1 file) ...... 85 kB
#     two_files: dots alone ............. 30 kB
#                dots + grey merged ..... 61 kB   (91 kB for the pair)
#     grey silhouette on its own ........ 59 kB
# So merging IS smaller than two images in one file, but writing two files
# costs more in total than one layered file, because the dots are stored twice.
# The grey is the expensive part either way: it covers the whole ROI, and image
# size follows AREA, not the number of bins. See RASTER_DPI if size matters
# more than resolution.
#   "grey_only"  One file per layer, the grey version only. The no-grey file is
#                not drawn at all, which also halves the rendering work.
#   "none"       No grey at all, one plain file per layer.
GREY_BG_MODE   = "grey_only"

# Per-kind override of GREY_BG_MODE. A kind listed here uses its own mode; any
# kind not listed uses GREY_BG_MODE above.
#
# The hand-picked maps are set to "none" because you asked for them without the
# grey silhouette. That also means their filenames carry no _greyROI suffix,
# since there is no second variant to distinguish them from. Delete the entry,
# or set it to "grey_only", to bring them back into line with everything else.
GREY_BG_MODE_BY_KIND = {
    "handpicked": "none",
}
GREY_BG_SUFFIX = "_greyROI"
# Kept even in grey_only mode, where it is strictly redundant, so a new file is
# instantly distinguishable from any stale no-grey file left over from an
# earlier run.
GREY_BG_SUFFIX_IN_GREY_ONLY = True

# Individual F / CL exports get no grey, as one follicle on a full-ROI grey
# field is mostly grey. True gives them the same treatment as the compartments.
GREY_BG_ON_FEATURE_LAYERS = False

# ---- ROI-slice restriction ------------------------------------------------
# Per-tissue plots keep only the Tissue-slices value(s) holding assigned
# ROI-Segmentation bins, so the unannotated serial section is not drawn and the
# axis limits close in on the annotated one.
RESTRICT_TO_ROI_SLICES = True

# ---- Illustrator layer exports --------------------------------------------
# Feature-only plots: no grey shadow, no axes, title kept so the layer is
# identifiable once imported. Saved without a tight bounding box so that every
# layer of one tissue shares an identical canvas and overlays exactly.
FEATURE_ONLY_TRANSPARENT = True   # False gives an opaque white background
FEATURE_ONLY_SHOW_TITLE  = True
FEATURE_ONLY_TITLE_FONT  = 12

# Also export each F / CL feature's halo on its own, with no bins under it, as
# <ID>-<comp><n>_halo. Same canvas and axis limits as every other layer, so the
# halo drops straight onto its overlaid version in Illustrator.
EXPORT_INDIVIDUAL_HALOS = True

# ---- Flat ROI map ---------------------------------------------------------
# A second ROI plot per tissue in which every compartment is ONE colour, its
# BASE_COLOURS value, with no per-feature lightness ramp. F1 and F7 are the
# same orange, CL1 and CL4 the same purple. Use it when the figure is making a
# four-compartment point and the individual feature identity is noise.
#
# The shaded version, <ID>-ROI-Segmentation, is still written as well. Both
# land in the same overview folder and share the tissue's axis limits and
# marker size, so they are interchangeable in a figure.
PLOT_FLAT_ROI   = True
FLAT_ROI_SUFFIX = "-ROI-flat"
# Halos are a fifth and sixth colour on the canvas, which defeats the point of
# a flat four-colour map, so they are off here. True draws them anyway.
FLAT_ROI_HALOS  = False

# ---- Shade ramp -----------------------------------------------------------
# The chosen BASE_COLOURS hex is the DARKEST shade; extra features get
# progressively lighter tints only. Going darker produced muddy browns for F.
SHADE_STEP     = 0.05   # lightness added per extra feature
SHADE_MAX_SPAN = 0.30   # total lightening cap, so late shades stay visible
SHADE_L_CAP    = 0.80   # absolute lightness ceiling, keeps shades off white

# ---- Region halo ----------------------------------------------------------
# A translucent area loosely tracing each F / CL feature, drawn as a heavily
# blurred contour so it follows the region without hugging every bin.
HALO_COMPARTMENTS = ("F", "CL")
HALO_ALPHA    = 0.8
# Each feature gets its OWN halo, drawn in its compartment's colour one shade
# lighter than the lightest tint used in the ramp (see halo_colour()).
#   F  -> #ecb596     CL -> #d3c5ed
# Drawn on top of the points for an opaque, histology-overlay look.
HALO_ON_TOP   = True
HALO_SIGMA    = 2.0         # blur in bins; rounds the outline only
HALO_DESPECKLE = 3          # binary closing/opening radius in bins
# Hard cap on how far the shaded area may extend past the outermost bin.
# One bin, so halos of adjacent features cannot run into each other.
HALO_EXPAND_UM = 4.0
HALO_BIN_UM   = 4.0
HALO_MIN_BINS = 15          # skip the halo for features smaller than this
HALO_MAX_CELLS = 12_000_000

# =============================================================================
# VALUE LAYERS: STIFFNESS AND GENE EXPRESSION ON THE SAME FEATURES
#
# Same crops as the Illustrator layers above (whole compartment, and each
# numbered F / CL feature), but coloured by a continuous value instead of a
# compartment colour. No axes, title only, PNG 600 dpi + SVG.
# =============================================================================

# ---- Stiffness ------------------------------------------------------------
PLOT_STIFFNESS = True

# Where the 50 um-assigned-to-4 um stiffness lives. Set to None if the column
# is already present in METADATA_TSV; the script checks that first anyway and
# only reads this file if it has to.
STIFFNESS_TSV = Path(
    r"Z:\Visium_Data\001_SpatialX\20260803_FinalAnnotatedMetadata"
    r"\metadata_with_nano.tsv"
)
STIFFNESS_COLS = ["Elasticity_l_Pa"]
# Plotted on a LINEAR scale. Put a column name in here only if you want that
# one log10-transformed; script 10 logs elasticity for correlation because rank
# statistics want a spread-out variable, but these maps are read as Pa.
STIFFNESS_LOG  = set()
STIFFNESS_CMAP = "YlOrRd"        # matches CMAP_STIFF in script 10
# Alternative in your own soft/stiff colours:
# STIFFNESS_CMAP = ["#56B4E9", "#F7F7B0", "#E69F00"]

# ---- Gene expression ------------------------------------------------------
# Fill this in once the correlation hits are known, e.g. ["Lhcgr", "Col1a1"].
# Empty list = the whole gene section is skipped and the expression matrix is
# never loaded.
GENES_TO_PLOT = [
    # S
    "Wnt4", "Cdkn1b", "Wnt10b", "Inhba", "Dpt", "Plod2",
    # CL
    "Kcnma1", "Jun", "Lgr5", "Ccn5", "S100g", "Ltbp1",
    # V
    "Srgn", "Cd34", "Fbln5",
]

# Restrict each gene to the compartment it was a hit in. None or {} = draw
# every gene in every compartment (the old behaviour).
GENE_COMPARTMENTS = {
    "Wnt4": ["S"], "Cdkn1b": ["S"], "Wnt10b": ["S"],
    "Inhba": ["S"], "Dpt": ["S"], "Plod2": ["S"],
    "Kcnma1": ["CL"], "Jun": ["CL"], "Lgr5": ["CL"],
    "Ccn5": ["CL"], "S100g": ["CL"], "Ltbp1": ["CL"],
    "Srgn": ["V"], "Cd34": ["V"], "Fbln5": ["V"],
}

# ---- Genes chosen from the compartment DEG tables -------------------------
# Instead of typing GENES_TO_PLOT by hand, take the compartment-specific
# Aged vs Young DEGs, keep only the ones that are also on the Mechanomics list,
# report how many survive in each direction, and ask how many of the top ones
# to plot. Answer the two prompts in PowerShell and the run carries on.
#
# When this is on it REPLACES GENES_TO_PLOT and GENE_COMPARTMENTS below.
DEG_GENE_SELECTION = True

DEG_DIR = Path(
    r"Z:\Visium_Data\001_SpatialX\20260803_FinalAnnotatedMetadata"
    r"\deg_gsea_outputs\1_compartment"
)
# compartment code -> the name used in the file, DEG_<name>_Aged_vs_Young.csv
DEG_COMPARTMENT_FILES = {
    "S":  "Stroma",
    "F":  "Follicle",
    "CL": "CL",
    "V":  "Vascular",
}
DEG_FILE_PATTERN = "DEG_{name}_Aged_vs_Young.csv"

# The Mechanomics set written by script 10.
MECHANOMICS_CSV = Path(
    r"Z:\Visium_Data\001_SpatialX\20260803_FinalAnnotatedMetadata"
    r"\pointlevel_expression_outputs_ageAdj\mechanomics_gene_set.csv"
)

DEG_PADJ_MAX = 0.05
# Kept only if log2FC > DEG_LFC_MIN or < -DEG_LFC_MIN, i.e. STRICTLY beyond the
# threshold, matching "more than 1 or less than -1". At 1.0 that is a two-fold
# change. Set to 0 to keep everything that passes DEG_PADJ_MAX.
DEG_LFC_MIN  = 1.0
DEG_RANK_BY  = "lfc"     # "lfc" ranks by effect size, "padj" by significance

# mechanomics_gene_set.csv carries an in_expression_matrix column. 1,724 of its
# 1,854 genes are in the matrix and 130 are not, so without this a gene can be
# picked as a top hit and then fail to plot, silently leaving you with fewer
# panels than you asked for. True drops those 130 at selection time instead.
# Ignored if the column is absent.
DEG_REQUIRE_IN_MATRIX = True

# True prints the counts and waits for two numbers. False, or a non-interactive
# session such as a piped or scheduled run, uses the defaults below without
# stopping. Enter at the prompt also accepts the default.
DEG_INTERACTIVE = True
DEG_TOP_UP   = 3
DEG_TOP_DOWN = 3
DEG_LIST_MAX = 25        # how many to print per direction, per compartment

# Column names are detected from these, in order, and the choice is printed so
# a wrong guess is visible rather than silent. Covers Seurat FindMarkers,
# DESeq2 and edgeR / limma output.
DEG_COL_CANDIDATES = {
    "gene": ["gene", "Gene", "genes", "symbol", "Symbol", "SYMBOL",
             "gene_name", "gene_symbol", "GeneSymbol", "X", "Unnamed: 0"],
    "lfc":  ["avg_log2FC", "log2FoldChange", "logFC", "avg_logFC",
             "log2FC", "Log2FC", "log2_fold_change"],
    "padj": ["p_val_adj", "padj", "FDR", "adj.P.Val", "qvalue", "q_value",
             "qval", "fdr", "BH", "p_adj"],
    "pval": ["p_val", "pvalue", "PValue", "P.Value", "pval", "p_value"],
}

# Force a column instead of detecting it. None = detect and report.
#
# Your DESeq2 tables carry BOTH log2FoldChange (shrunken, from lfcShrink) and
# log2FoldChange_mle (the raw maximum-likelihood estimate). Detection takes the
# shrunken one, which is the right default for ranking: shrinkage exists
# precisely to pull the inflated fold changes of low-count, high-variance genes
# back toward zero, so ranking on the MLE would push noisy low-expression genes
# to the top of your plot list. Set DEG_LFC_COLUMN = "log2FoldChange_mle" if
# you specifically want the unshrunken estimate.
DEG_GENE_COLUMN = None
DEG_LFC_COLUMN  = None
DEG_PADJ_COLUMN = None

# Column pairs that mean "shrunken estimate" / "unshrunken MLE". Used only to
# print a note when both are present, so the choice is never silent.
DEG_SHRUNK_MLE_PAIRS = [("log2FoldChange", "log2FoldChange_mle")]

EXPR_DIR = Path(
    r"Z:\Visium_Data\001_SpatialX\20260803_FinalAnnotatedMetadata"
    r"\expression_export"
)
EXPR_MATRIX_FILE   = "step9_expression_matrix.mtx"
EXPR_BARCODES_FILE = "step9_expression_barcodes.txt"
EXPR_FEATURES_FILE = "step9_expression_features.txt"
EXPR_META_FILE     = "step9_expression_metadata_matched.tsv"

# ---- Gene expression colour ramp ------------------------------------------
# Any number of hex stops, interpolated in order from lowest to highest
# expression. THIS is the only place to change the gene colours.
#
# Why it changed. The old two-stop ramp #FFB0D7 -> #FD007E spans HLS lightness
# 85 down to 50, a span of 35, and both ends are saturated pink, so a low bin
# and a high bin differ mostly in hue rather than in darkness. On a printed
# figure that reads as one flat pink field. The ramp below runs 97 -> 22, a
# span of 76, which is 2.2x the contrast, and it still passes through your
# #FD007E at the midpoint so the overall look is unchanged.
#
# Alternatives, measured the same way:
#   near-white -> dark plum, no hot pink   span 70
#     GENE_CMAP_STOPS = ["#FFF0F7", "#8C0045"]
#   maximum contrast, top end close to black  span 83
#     GENE_CMAP_STOPS = ["#FFF2F8", "#F5178C", "#4A0022"]
GENE_CMAP_STOPS = ["#FFF2F8", "#FD007E", "#6E0033"]

# Kept only so an older config that sets these still runs. If either is not
# None they override GENE_CMAP_STOPS with a plain two-stop ramp.
GENE_CMAP_LO = None
GENE_CMAP_HI = None

GENE_ZERO_AS_BLANK = True

# ---- Shared value-layer settings ------------------------------------------
# "global" puts every tissue on one colour scale, so young and aged are
# directly comparable. "tissue" rescales per ovary, which shows within-tissue
# structure better but makes cross-tissue comparison invalid.
VALUE_SCALE_SCOPE = "global"        # "global" or "tissue"
VALUE_CLIP_PCT    = (2, 98)         # percentile clip, kills single-bin outliers
MISSING_VALUE_COLOUR = "#ececec"    # only used if DRAW_MISSING_BINS is True
DRAW_MISSING_BINS = False           # bins with no stiffness are left blank
SAVE_COLOURBARS   = True            # one small separate file per value

# Barcode column candidates, used to join stiffness / expression onto the
# segmentation table.
BARCODE_COL_CANDIDATES = ["Unique Barcodes", "barcode_full", "Barcodes",
                          "unique_barcodes", "Barcode"]


# =============================================================================
# HELPERS
# =============================================================================

def parse_seg(label):
    """
    "Y1-F-3"    → ("Y1",   "F",  3)
    "Y1-CL-2"   → ("Y1",   "CL", 2)
    "Y1-S"      → ("Y1",   "S",  None)
    "A3-2-CL-1" → ("A3-2", "CL", 1)
    "Unassigned" → (None, None, None)
    """
    s = str(label).strip()
    if s in SKIP:
        return None, None, None
    m = re.match(r'^(.+?)-(CL|F|S|V)(?:-(\d+))?$', s)
    if not m:
        return None, None, None
    return m.group(1), m.group(2), (int(m.group(3)) if m.group(3) else None)


def generate_shades(base_hex, n):
    """n shades from dark to light, preserving hue and saturation."""
    if n == 0:
        return []
    if n == 1:
        return [base_hex]
    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_hex))
    # base_hex is the DARKEST shade; only lighten from here
    span  = min(SHADE_MAX_SPAN, SHADE_STEP * (n - 1))
    l_max = min(SHADE_L_CAP, l + span)
    return [mcolors.to_hex(colorsys.hls_to_rgb(h, float(lv), s))
            for lv in np.linspace(l, l_max, n)]


# comp -> lightness ramp indexed by (feature number - 1). Built once in main()
# over every tissue, so CL3 is the same purple everywhere.
SHADE_MAPS = {}


def build_shade_maps(df):
    """Ramp per compartment over the highest feature NUMBER seen anywhere."""
    for comp in HALO_COMPARTMENTS:
        nums = [n for v in pd.Series(df[COL_SEG]).dropna().unique()
                for (_, c, n) in [parse_seg(v)]
                if c == comp and n is not None]
        if nums:
            SHADE_MAPS[comp] = generate_shades(BASE_COLOURS[comp], max(nums))
            print(f"  {comp}: feature numbers up to {max(nums)}, "
                  f"{len(SHADE_MAPS[comp])} shades "
                  f"({SHADE_MAPS[comp][0]} -> {SHADE_MAPS[comp][-1]})")


def shades_for(comp, nums):
    """
    Colours for these feature numbers.

    Under SHADE_SCOPE = "global" the colour is looked up BY FEATURE NUMBER in a
    ramp built across every tissue, so CL3 is one specific purple no matter how
    many CLs its ovary happens to have. Under "tissue" the ramp is rebuilt from
    the local count, which is the old behaviour: only CL1 was comparable
    between tissues, every later shade drifted.
    """
    if SHADE_SCOPE == "global" and comp in SHADE_MAPS:
        ramp = SHADE_MAPS[comp]
        return [ramp[min(int(n), len(ramp)) - 1] for n in nums]
    return generate_shades(BASE_COLOURS[comp], len(nums))


def marker_s(x_range, y_range):
    """Placeholder. The real size is set by fit_bin_markers() at save time."""
    return 1.0


def measure_bin_pitch(x, y):
    """Median nearest-neighbour distance, i.e. the bin centre-to-centre pitch."""
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return float(BIN_PITCH_FALLBACK_UM)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n < 8:
        return float(BIN_PITCH_FALLBACK_UM)
    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    q = pts
    if n > BIN_PITCH_SAMPLE:
        idx = np.random.default_rng(0).choice(n, BIN_PITCH_SAMPLE,
                                              replace=False)
        q = pts[idx]
    d, _ = tree.query(q, k=2)
    pitch = float(np.median(d[:, 1]))
    if not np.isfinite(pitch) or pitch <= 0:
        return float(BIN_PITCH_FALLBACK_UM)
    return pitch


def axes_points_per_um(fig, ax):
    fig.canvas.draw()
    bb = ax.get_window_extent()
    w_pts = bb.width / fig.dpi * 72.0
    h_pts = bb.height / fig.dpi * 72.0
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (w_pts / max(abs(x1 - x0), 1e-9),
            h_pts / max(abs(y1 - y0), 1e-9))


def fit_bin_markers(fig):
    """Resize every registered bin scatter so one marker covers one bin pitch."""
    for ax in fig.axes:
        cols = getattr(ax, "_bin_collections", None)
        if not cols:
            continue
        offs = [np.asarray(c.get_offsets()) for c in cols]
        offs = [o for o in offs if o.size]
        if not offs:
            continue
        allpts = np.vstack(offs)
        pitch = measure_bin_pitch(allpts[:, 0], allpts[:, 1])
        ppu = min(axes_points_per_um(fig, ax))
        side = pitch * BIN_OVERLAP * ppu
        size = max(side ** 2, 1e-6)
        for c in cols:
            c.set_sizes(np.full(len(c.get_offsets()), size))


def new_fig():
    """Fixed-size canvas for all per-tissue plots."""
    return plt.subplots(figsize=(CANVAS_W, CANVAS_H))


def apply_style(ax, xlim, ylim, title, handles):
    """
    Shared formatting for the overview plots — fixed axis limits.

    With OVERVIEW_SHOW_AXES = False the spines, ticks and axis labels are
    switched off and the axes patch is made transparent. set_axis_off() only
    affects the frame: ax.title and the legend are separate artists and both
    still draw, which is what keeps these identifiable and readable.
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    if OVERVIEW_SHOW_AXES:
        ax.set_facecolor("none" if TRANSPARENT_BACKGROUND else "white")
        ax.set_xlabel("x (µm)", fontsize=10)
        ax.set_ylabel("y (µm)", fontsize=10)
        ax.tick_params(labelsize=8)
    else:
        ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    if handles:
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
            fontsize=7,
            markerscale=1,          # square markers are already sized in patch()
            handlelength=1.0,       # keep the swatch square, not a wide bar
            handleheight=1.0,
            handletextpad=0.7,
            labelspacing=0.55,
            frameon=LEGEND_FRAME,
            framealpha=0.9,
        )


def age_group_of(ident):
    """
    "Y1", "Y4-2", "A3", "A3-1" -> "Young" / "Aged".

    Only the leading letter is read. Anything that is not Y or A returns the
    ungrouped folder rather than defaulting to one age, so a tissue named
    outside the convention shows up as a stray folder instead of quietly
    joining the wrong group.
    """
    s = str(ident).strip().upper()
    if s.startswith("Y"):
        return DIR_YOUNG
    if s.startswith("A"):
        return DIR_AGED
    return DIR_UNGROUPED


_DIRS_MADE = set()


def _safe_name(s):
    """A gene symbol usable as a folder name on Windows."""
    return re.sub(r'[<>:"/\\|?*]', "_", str(s).strip()) or "unnamed"


def out_dir_for(ident, kind, level=None, gene=None):
    """
    Directory for one image, created on demand.

    ident  OvaryID or Tissue-slices value, used only for the age split.
           None puts the file in OUTPUT_DIR itself, which is what the colour
           bars and the all-capture-areas map want because they are not
           specific to one age.
    kind   overview | features | stiffness | gene
    level  "compartment" for a whole-compartment export, "feature" for a
           numbered F / CL one, None for the overview maps.
    """
    # Gene layers live in their own tree, subdivided per GENE_GROUP_BY rather
    # than by age, since a gene's young and aged panels are compared to each
    # other and splitting them by age would put them in different folders.
    if kind == "handpicked":
        d = OUTPUT_DIR / HANDPICKED_DIR
        if d not in _DIRS_MADE:
            d.mkdir(parents=True, exist_ok=True)
            _DIRS_MADE.add(d)
        return d

    if kind == "gene":
        root = OUTPUT_DIR / GENES_ROOT_DIR
        if GENE_GROUP_BY == "tissue" and ident:
            d = root / _safe_name(ident)
        elif GENE_GROUP_BY == "gene" and gene:
            d = root / _safe_name(gene)
        else:
            d = root
        if d not in _DIRS_MADE:
            d.mkdir(parents=True, exist_ok=True)
            _DIRS_MADE.add(d)
        return d

    if not ORGANISE_OUTPUT or ident is None:
        d = OUTPUT_DIR
    else:
        if SPLIT_COMPARTMENT_AND_FEATURES:
            key = (kind, None if kind == "overview" else level)
            sub = KIND_TO_SUBDIR_SPLIT.get(key)
            if sub is None:
                raise ValueError(
                    f"unknown output route {key!r}; expected one of "
                    f"{sorted(KIND_TO_SUBDIR_SPLIT)}")
        else:
            sub = KIND_TO_SUBDIR.get(kind)
            if sub is None:
                raise ValueError(f"unknown output kind {kind!r}; "
                                 f"expected one of {sorted(KIND_TO_SUBDIR)}")
        d = OUTPUT_DIR / age_group_of(ident) / sub
    if d not in _DIRS_MADE:
        d.mkdir(parents=True, exist_ok=True)
        _DIRS_MADE.add(d)
    return d


def _rel(path):
    """Path shown in the log, relative to OUTPUT_DIR so lines stay short."""
    try:
        return str(path.relative_to(OUTPUT_DIR))
    except ValueError:
        return str(path)


def save_all(fig, stem, out_dir=None, transparent=None):
    """
    Save PNG, TIFF, SVG. out_dir defaults to OUTPUT_DIR itself.

    The SVG is written with dpi = RASTER_DPI. That does not change the SVG's
    dimensions, which come from figsize alone; it only sets the resolution of
    any artist marked rasterized, i.e. the dots.
    """
    d = out_dir if out_dir is not None else OUTPUT_DIR
    tr = TRANSPARENT_BACKGROUND if transparent is None else transparent
    fc = "none" if tr else "white"
    fit_bin_markers(fig)
    written = []
    for ext, kwargs in (
        ("png",  {"dpi": PNG_DPI}),
        ("tiff", {"dpi": TIFF_DPI}),
        ("svg",  {"dpi": RASTER_DPI}),
    ):
        if ext not in EXPORT_FORMATS:
            continue
        fig.savefig(d / f"{stem}.{ext}",
                    bbox_inches="tight", facecolor=fc,
                    edgecolor="none", transparent=tr, **kwargs)
        written.append("." + ext)
    plt.close(fig)
    print(f"    saved: {_rel(d / stem)}  ({' / '.join(written)})")


def apply_style_bare(ax, xlim, ylim, title):
    """
    Same framing as apply_style but with the axes switched off entirely.

    set_axis_off() removes the spines, ticks and tick labels; ax.title is a
    separate artist and still draws, which is what keeps the layer
    identifiable in Illustrator.
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if FEATURE_ONLY_SHOW_TITLE:
        ax.set_title(title, fontsize=FEATURE_ONLY_TITLE_FONT,
                     fontweight="bold", pad=8)


def save_png_svg(fig, stem, transparent=None, out_dir=None, formats=None):
    """
    PNG + SVG only, for the Illustrator layer exports.

    bbox_inches is deliberately NOT "tight". Tight cropping trims each figure
    to whatever it happens to contain, so a single-follicle layer would come
    out cropped differently from the stroma layer and the two would not line up
    when stacked. Keeping the full canvas, with identical figsize and identical
    axis limits, means every layer of one tissue registers exactly.
    """
    d = out_dir if out_dir is not None else OUTPUT_DIR
    tr = (TRANSPARENT_BACKGROUND or FEATURE_ONLY_TRANSPARENT) \
        if transparent is None else transparent
    want = EXPORT_FORMATS if formats is None else formats
    fit_bin_markers(fig)
    written = []
    for ext, kwargs in (("png",  {"dpi": PNG_DPI}),
                        ("tiff", {"dpi": TIFF_DPI}),
                        ("svg",  {"dpi": RASTER_DPI})):
        if ext not in want:
            continue
        fig.savefig(d / f"{stem}.{ext}",
                    bbox_inches=None,
                    facecolor="none" if tr else "white",
                    edgecolor="none",
                    transparent=tr, **kwargs)
        written.append("." + ext)
    plt.close(fig)
    print(f"    saved: {_rel(d / stem)}  ({' / '.join(written)})")


def scatter_sq(ax, x, y, colour, s, zorder=3):
    """Square-marker scatter, no edge lines."""
    sc = ax.scatter(x, y, c=colour, s=s, marker="s",
                    linewidths=0, zorder=zorder,
                    rasterized=RASTERIZE_DOTS)
    if not hasattr(ax, "_bin_collections"):
        ax._bin_collections = []
    ax._bin_collections.append(sc)
    return sc


def halo_colour(base_hex):
    """One lightness step beyond the lightest tint the ramp can produce."""
    h, l, s_ = colorsys.rgb_to_hls(*mcolors.to_rgb(base_hex))
    lightest = min(SHADE_L_CAP, l + SHADE_MAX_SPAN)
    return mcolors.to_hex(
        colorsys.hls_to_rgb(h, min(0.92, lightest + SHADE_STEP), s_))


def patch(colour, label):
    """Square legend marker rather than the default wide rectangle."""
    return mlines.Line2D([], [], color="none", marker="s", markersize=7,
                          markerfacecolor=colour, markeredgecolor="none",
                          linestyle="none", label=label)


def halo(ax, x, y, colour, zorder):
    """Translucent area loosely tracing a feature.

    Rasters the bins onto their 4 um lattice, despeckles, blurs heavily, then
    fills the 0.5 contour. The heavy blur is deliberate: it produces a loose
    region estimate rather than an outline that follows every bin edge.

    Returns True if a halo was actually drawn. Several guards below bail out
    silently (feature too small, mask never reaches the 0.5 contour), and the
    halo-only exports need to know so they do not write a blank file.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < HALO_MIN_BINS:
        return False
    pad = int(np.ceil(3 * HALO_SIGMA)) + int(round(HALO_EXPAND_UM / HALO_BIN_UM)) + 2
    x0, y0 = x.min(), y.min()
    ix = np.rint((x - x0) / HALO_BIN_UM).astype(int)
    iy = np.rint((y - y0) / HALO_BIN_UM).astype(int)
    nx = int(ix.max()) + 1 + 2 * pad
    ny = int(iy.max()) + 1 + 2 * pad
    if nx * ny > HALO_MAX_CELLS or nx < 3 or ny < 3:
        return False
    m = np.zeros((ny, nx), dtype=bool)
    m[iy + pad, ix + pad] = True
    if HALO_DESPECKLE > 0:
        # close small internal gaps, then drop isolated stray bins
        m = ndimage.binary_closing(m, iterations=HALO_DESPECKLE)
        m = ndimage.binary_opening(m, iterations=max(1, HALO_DESPECKLE - 1))
    grow = int(round(HALO_EXPAND_UM / HALO_BIN_UM))
    allowed = ndimage.binary_dilation(m, iterations=grow) if grow > 0 else m
    m = m.astype(float)
    if HALO_SIGMA > 0:
        m = ndimage.gaussian_filter(m, sigma=HALO_SIGMA)
    # Blurring can push the 0.5 contour outward at convex corners. Clamp it to
    # the bins plus HALO_EXPAND_UM so one feature's halo can never reach
    # another's, however close they sit.
    m = np.where(allowed, m, 0.0)
    if m.max() < 0.5:
        return False
    xs = x0 + (np.arange(nx) - pad) * HALO_BIN_UM
    ys = y0 + (np.arange(ny) - pad) * HALO_BIN_UM
    cs = ax.contourf(xs, ys, m, levels=[0.5, m.max() + 1.0],
                     colors=[colour], alpha=HALO_ALPHA,
                     zorder=zorder, antialiased=True)
    if RASTERIZE_HALOS:
        cs.set_rasterized(True)
    return True


# =============================================================================
# PER-TISSUE PLOTS
# =============================================================================

def _tissue_extent(tissue_df, span=None):
    """
    Axis limits for all sub-plots of this tissue.

    With span = (width_um, height_um) the window is that fixed size, centred on
    this tissue's middle, so every tissue is drawn at the same micrometres per
    inch and feature sizes are directly comparable between ovaries. Without it,
    limits hug this tissue alone and small ovaries come out magnified.
    """
    xs = tissue_df[COL_X]
    ys = tissue_df[COL_Y]
    if span is not None:
        w, h = span
        cx = (xs.min() + xs.max()) / 2.0
        cy = (ys.min() + ys.max()) / 2.0
        return (cx - w / 2.0, cx + w / 2.0), (cy - h / 2.0, cy + h / 2.0)
    pad_x = (xs.max() - xs.min()) * EXTENT_PAD
    pad_y = (ys.max() - ys.min()) * EXTENT_PAD
    return ((xs.min() - pad_x, xs.max() + pad_x),
            (ys.min() - pad_y, ys.max() + pad_y))


def _shadow(ax, tissue_df, s):
    """Draw all tissue bins as light grey in the background."""
    scatter_sq(ax, tissue_df[COL_X].values, tissue_df[COL_Y].values,
               SHADOW_COLOUR, s, zorder=1)


def grey_bg_frame(tissue_df):
    """
    The bins that make up the grey silhouette, per GREY_BG_SCOPE.

    "roi" is every bin carrying an assigned ROI-Segmentation label, so the
    silhouette is the four compartments merged and nothing else. "tissue" is
    every bin in the frame, which by this point is already restricted to the
    ROI-bearing slice, so it is the whole section.
    """
    if GREY_BG_SCOPE == "tissue":
        return tissue_df
    return tissue_df[tissue_df[COL_SEG].apply(
        lambda v: parse_seg(v)[1] is not None)]


def new_layer_fig(xlim, ylim, s, bg_df=None, merged=True):
    """
    Canvas for a layer export, optionally with a grey silhouette behind it.

    merged=True  puts the grey in the SAME Axes as the layer, at zorder 1.
                 Matplotlib merges consecutive rasterised artists into one
                 embedded image, so the pair arrives as a single flattened
                 picture. Smaller, but the grey cannot be moved on its own.
    merged=False puts the grey in a SECOND Axes at exactly ax's position and
                 one step lower in zorder, which forces a separate <image> and
                 keeps it movable. Taking the position from ax.get_position()
                 rather than hard-coding a rect is what keeps these registering
                 with every other export.
    """
    fig, ax = new_fig()
    if bg_df is None or not len(bg_df):
        return fig, ax
    if merged:
        scatter_sq(ax, bg_df[COL_X].values, bg_df[COL_Y].values,
                   SHADOW_COLOUR, s, zorder=1)
    else:
        ax_bg = fig.add_axes(ax.get_position(), label="grey_background",
                             zorder=ax.get_zorder() - 1)
        ax_bg.set_xlim(xlim)
        ax_bg.set_ylim(ylim)
        ax_bg.set_aspect("equal")
        ax_bg.set_axis_off()
        ax_bg.patch.set_visible(False)
        scatter_sq(ax_bg, bg_df[COL_X].values, bg_df[COL_Y].values,
                   SHADOW_COLOUR, s, zorder=1)
    return fig, ax


def grey_mode_for(kind):
    """GREY_BG_MODE, unless this kind has its own override."""
    return GREY_BG_MODE_BY_KIND.get(kind, GREY_BG_MODE)


def emit_layer(draw, stem, out_dir, xlim, ylim, title, bg_df, s,
               formats=None, mode=None):
    """
    Write a compartment layer once per GREY_BG_MODE variant.

    draw(ax) adds the layer's own content. It is called fresh for each variant
    rather than once into a reused figure, because a matplotlib artist belongs
    to exactly one Axes and cannot be moved between figures.
    """
    mode = GREY_BG_MODE if mode is None else mode
    have_bg = bg_df is not None and len(bg_df) > 0 and mode != "none"

    if mode == "two_files" and have_bg:
        variants = [(None, stem, True),
                    (bg_df, stem + GREY_BG_SUFFIX, True)]
    elif mode == "grey_only" and have_bg:
        variants = [(bg_df,
                     stem + (GREY_BG_SUFFIX if GREY_BG_SUFFIX_IN_GREY_ONLY
                             else ""),
                     True)]
    elif mode == "layered" and have_bg:
        variants = [(bg_df, stem, False)]
    else:
        variants = [(None, stem, True)]

    for bg, out_stem, merged in variants:
        fig, ax = new_layer_fig(xlim, ylim, s, bg_df=bg, merged=merged)
        draw(ax)
        apply_style_bare(ax, xlim, ylim, title)
        save_png_svg(fig, out_stem, out_dir=out_dir, formats=formats)


def plot_single_compartment(tissue_df, tissue_id, comp,
                             xlim, ylim, s):
    base = BASE_COLOURS[comp]
    mask = tissue_df[COL_SEG].apply(lambda v: parse_seg(v)[1] == comp)
    sub  = tissue_df[mask].copy()
    if sub.empty:
        return

    fig, ax = new_fig()
    _shadow(ax, tissue_df, s)
    handles = []

    if comp in ("S", "V"):
        scatter_sq(ax, sub[COL_X].values, sub[COL_Y].values, base, s)
        handles.append(patch(base, comp))
    else:
        sub["_n"] = sub[COL_SEG].apply(lambda v: parse_seg(v)[2])
        nums   = sorted(sub["_n"].dropna().unique())
        shades = shades_for(comp, nums)
        halo_z = 4 if HALO_ON_TOP else 2
        for num, col in zip(nums, shades):
            rows = sub[sub["_n"] == num]
            if comp in HALO_COMPARTMENTS:
                halo(ax, rows[COL_X].values, rows[COL_Y].values,
                     halo_colour(base), halo_z)
            scatter_sq(ax, rows[COL_X].values, rows[COL_Y].values, col, s)
            handles.append(patch(col, f"{comp}{int(num)}"))

    apply_style(ax, xlim, ylim, f"{tissue_id}  ·  {comp}", handles)
    save_all(fig, f"{tissue_id}-{comp}",
             out_dir=out_dir_for(tissue_id, "overview"))


def plot_tissue_combined(tissue_df, tissue_id, xlim, ylim, s):
    assigned = tissue_df[tissue_df[COL_SEG].apply(
        lambda v: parse_seg(v)[1] is not None)].copy()
    if assigned.empty:
        return

    fig, ax = new_fig()
    _shadow(ax, tissue_df, s)
    handles = []

    for comp in ("F", "CL", "S", "V"):
        base  = BASE_COLOURS[comp]
        cmask = assigned[COL_SEG].apply(lambda v: parse_seg(v)[1] == comp)
        sub   = assigned[cmask].copy()
        if sub.empty:
            continue

        if comp in ("S", "V"):
            scatter_sq(ax, sub[COL_X].values, sub[COL_Y].values, base, s)
            handles.append(patch(base, comp))
        else:
            sub["_n"] = sub[COL_SEG].apply(lambda v: parse_seg(v)[2])
            nums   = sorted(sub["_n"].dropna().unique())
            shades = shades_for(comp, nums)
            halo_z = 4 if HALO_ON_TOP else 2
            for num, col in zip(nums, shades):
                rows = sub[sub["_n"] == num]
                if comp in HALO_COMPARTMENTS:
                    halo(ax, rows[COL_X].values, rows[COL_Y].values,
                     halo_colour(base), halo_z)
                scatter_sq(ax, rows[COL_X].values, rows[COL_Y].values,
                           col, s)
                handles.append(patch(col, f"{comp}{int(num)}"))

    apply_style(ax, xlim, ylim, f"{tissue_id}  ·  ROI Segmentation", handles)
    save_all(fig, f"{tissue_id}-ROI-Segmentation",
             out_dir=out_dir_for(tissue_id, "overview"))


def plot_tissue_flat(tissue_df, tissue_id, xlim, ylim, s):
    """
    <ID>-ROI-flat: the four compartments, four colours, no feature shades.

    Same source column, same axis limits and same marker size as
    plot_tissue_combined, so this and the shaded version are drawn at identical
    scale and can be swapped in a figure. The only difference is that colour is
    looked up by compartment rather than by feature number, so every follicle
    is one orange and every corpus luteum one purple.

    Painted background first (S, V) and features last (F, CL). With correct
    data that makes no difference, because ROI-Segmentation gives each bin
    exactly one label so no two compartments share a coordinate. It only
    matters if a bin were ever double-labelled, and in that case losing a
    follicle under the stroma is the worse failure. The legend is still built
    in F, CL, S, V order so it reads the same for every tissue, including one
    that is missing a compartment.
    """
    assigned = tissue_df[tissue_df[COL_SEG].apply(
        lambda v: parse_seg(v)[1] is not None)].copy()
    if assigned.empty:
        return

    assigned["_c"] = assigned[COL_SEG].apply(lambda v: parse_seg(v)[1])

    fig, ax = new_fig()
    _shadow(ax, tissue_df, s)
    handles = []
    halo_z = 4 if HALO_ON_TOP else 2

    for comp in ("S", "V", "F", "CL"):          # background first, features last
        sub = assigned[assigned["_c"] == comp]
        if sub.empty:
            continue
        base = BASE_COLOURS[comp]
        if FLAT_ROI_HALOS and comp in HALO_COMPARTMENTS:
            # still per feature, otherwise adjacent follicles merge into one blob
            for num in sorted(sub[COL_SEG].apply(
                    lambda v: parse_seg(v)[2]).dropna().unique()):
                rows = sub[sub[COL_SEG].apply(lambda v: parse_seg(v)[2]) == num]
                halo(ax, rows[COL_X].values, rows[COL_Y].values,
                     halo_colour(base), halo_z)
        scatter_sq(ax, sub[COL_X].values, sub[COL_Y].values, base, s)

    handles = [patch(BASE_COLOURS[c], c) for c in ("F", "CL", "S", "V")
               if (assigned["_c"] == c).any()]

    apply_style(ax, xlim, ylim,
                f"{tissue_id}  ·  ROI compartments", handles)
    save_all(fig, f"{tissue_id}{FLAT_ROI_SUFFIX}",
             out_dir=out_dir_for(tissue_id, "overview"))


# =============================================================================
# ILLUSTRATOR LAYER EXPORTS  (no shadow, no axes, title only)
# =============================================================================

def plot_compartment_only(tissue_df, tissue_id, comp, xlim, ylim, s):
    """
    <ID>_Fonly / _CLonly / _Sonly / _Vonly

    The compartment alone on a transparent canvas, at the tissue's shared axis
    limits. Shades and halos are identical to the combined plot, because both
    derive from the same sorted feature numbers.

    Under GREY_BG_MODE = "two_files" this also writes <ID>_<comp>only_greyROI,
    the same content over the grey ROI silhouette.
    """
    base = BASE_COLOURS[comp]
    mask = tissue_df[COL_SEG].apply(lambda v: parse_seg(v)[1] == comp)
    sub  = tissue_df[mask].copy()
    if sub.empty:
        return

    if comp not in ("S", "V"):
        sub["_n"] = sub[COL_SEG].apply(lambda v: parse_seg(v)[2])
        nums   = sorted(sub["_n"].dropna().unique())
        shades = shades_for(comp, nums)
    halo_z = 4 if HALO_ON_TOP else 2

    def draw(ax):
        if comp in ("S", "V"):
            scatter_sq(ax, sub[COL_X].values, sub[COL_Y].values, base, s)
            return
        for num, col in zip(nums, shades):
            rows = sub[sub["_n"] == num]
            if comp in HALO_COMPARTMENTS:
                halo(ax, rows[COL_X].values, rows[COL_Y].values,
                     halo_colour(base), halo_z)
            scatter_sq(ax, rows[COL_X].values, rows[COL_Y].values, col, s)

    emit_layer(draw, f"{tissue_id}_{comp}only",
               out_dir_for(tissue_id, "features", "compartment"),
               xlim, ylim, f"{tissue_id}  ·  {comp} only",
               grey_bg_frame(tissue_df), s,
               mode=grey_mode_for("features"))


def plot_individual_features(tissue_df, tissue_id, comp, xlim, ylim, s):
    """
    <ID>-F1, <ID>-F2, <ID>-CL1 …  one file per numbered feature.

    Each feature keeps the exact shade it has in the combined plot: the ramp is
    built over the full sorted list of feature numbers first, then indexed,
    rather than regenerated per feature.
    """
    base = BASE_COLOURS[comp]
    mask = tissue_df[COL_SEG].apply(lambda v: parse_seg(v)[1] == comp)
    sub  = tissue_df[mask].copy()
    if sub.empty:
        return

    sub["_n"] = sub[COL_SEG].apply(lambda v: parse_seg(v)[2])
    nums   = sorted(sub["_n"].dropna().unique())
    if not nums:
        return
    shades = shades_for(comp, nums)
    halo_z = 4 if HALO_ON_TOP else 2
    fdir = out_dir_for(tissue_id, "features", "feature")
    bg = grey_bg_frame(tissue_df) if GREY_BG_ON_FEATURE_LAYERS else None

    for num, col in zip(nums, shades):
        rows = sub[sub["_n"] == num]
        if rows.empty:
            continue

        # overlaid: halo plus the bins
        fig, ax = new_layer_fig(xlim, ylim, s, bg_df=bg)
        if comp in HALO_COMPARTMENTS:
            halo(ax, rows[COL_X].values, rows[COL_Y].values,
                 halo_colour(base), halo_z)
        scatter_sq(ax, rows[COL_X].values, rows[COL_Y].values, col, s)
        apply_style_bare(ax, xlim, ylim,
                         f"{tissue_id}  ·  {comp}{int(num)}")
        save_png_svg(fig, f"{tissue_id}-{comp}{int(num)}", out_dir=fdir)

        # halo alone, same canvas and limits so it registers on the above
        if EXPORT_INDIVIDUAL_HALOS and comp in HALO_COMPARTMENTS:
            fig, ax = new_layer_fig(xlim, ylim, s, bg_df=bg)
            drawn = halo(ax, rows[COL_X].values, rows[COL_Y].values,
                         halo_colour(base), halo_z)
            if not drawn:
                plt.close(fig)
                print(f"    skipped: {tissue_id}-{comp}{int(num)}_halo  "
                      f"(no halo drawn: {len(rows)} bins, "
                      f"HALO_MIN_BINS = {HALO_MIN_BINS})")
            else:
                apply_style_bare(ax, xlim, ylim,
                                 f"{tissue_id}  ·  {comp}{int(num)} halo")
                save_png_svg(fig, f"{tissue_id}-{comp}{int(num)}_halo",
                             out_dir=fdir)


# =============================================================================
# VALUE LAYERS: LOADING
# =============================================================================

# =============================================================================
# JOIN KEYS
#
# The two sides of a join must be built the SAME way. The previous version
# chose a column independently on each frame, which is how the gene layers came
# out empty: the segmentation TSV has "Unique Barcodes" (SpatialX long form,
# 'HTO-01-4um - <hash> - s_004um_00000_00796-1'), the step9 expression metadata
# does not and fell through to "barcode_full" ('s_004um_00144_00875-1.CA_01').
# Two different formats, zero overlap, every value NaN, nothing drawn.
#
# Every strategy below is now evaluated on BOTH frames, the overlap of each is
# reported, and the best is used. A join that matches almost nothing is a hard
# error rather than a warning, because a silent empty result is exactly what
# cost us a run.
# =============================================================================

CAPTURE_AREA_CANDIDATES = ["CaptureArea", "samples", "sample", "Sample",
                           "capture_area", "CA"]

# A join is rejected if fewer than this fraction of the RIGHT-hand table's
# barcodes find a match. The right table is the smaller one (the expression
# export, or the nanoindentation metadata), so this should be near 1.
JOIN_MIN_OVERLAP_FRAC = 0.5


def _key_barcode_full(frame):
    """The 'Barcodes.CaptureArea' convention, however the frame stores it."""
    if "barcode_full" in frame.columns:
        return frame["barcode_full"].astype(str).str.strip(), "barcode_full"
    if "Barcodes" in frame.columns:
        for sc in CAPTURE_AREA_CANDIDATES:
            if sc in frame.columns:
                return (frame["Barcodes"].astype(str).str.strip() + "." +
                        frame[sc].astype(str).str.strip()), f"Barcodes+{sc}"
    return None, None


def _key_named(col):
    def f(frame):
        if col in frame.columns:
            return frame[col].astype(str).str.strip(), col
        return None, None
    return f


KEY_STRATEGIES = [
    ("barcode_full",    _key_barcode_full),
    ("Unique Barcodes", _key_named("Unique Barcodes")),
    ("Barcodes",        _key_named("Barcodes")),
]


def choose_join_key(left, right, left_name="left", right_name="right"):
    """
    Try every strategy on both frames, report each, return the best.

    Returns (left_key, right_key, label). Raises if nothing overlaps.
    """
    print(f"  Join key search: {left_name} <- {right_name}")
    best = None
    for label, fn in KEY_STRATEGIES:
        lk, lcol = fn(left)
        rk, rcol = fn(right)
        if lk is None or rk is None:
            missing = left_name if lk is None else right_name
            print(f"    {label:16s} not available on {missing}")
            continue
        lset, rset = set(lk), set(rk)
        ov = len(lset & rset)
        frac = ov / max(len(rset), 1)
        uniq = "unique" if lk.nunique() == len(lk) else "NOT UNIQUE"
        print(f"    {label:16s} {lcol} / {rcol}: {ov:,} matched "
              f"({frac:.1%} of {right_name}); {left_name} key is {uniq}")
        if best is None or ov > best[3]:
            best = (lk, rk, label, ov, frac, lk.nunique() == len(lk))

    if best is None:
        raise ValueError(
            f"No shared barcode column between {left_name} and {right_name}. "
            f"{left_name} has {[c for c in left.columns if 'arcode' in c]}; "
            f"{right_name} has {[c for c in right.columns if 'arcode' in c]}.")

    lk, rk, label, ov, frac, is_unique = best
    if frac < JOIN_MIN_OVERLAP_FRAC:
        raise ValueError(
            f"Best join ({label}) matched only {ov:,} barcodes, {frac:.1%} of "
            f"{right_name}, below JOIN_MIN_OVERLAP_FRAC = "
            f"{JOIN_MIN_OVERLAP_FRAC}. Refusing to continue with a join this "
            f"poor; it would produce blank or wrong maps. Check the barcode "
            f"formats printed above.")
    if not is_unique:
        print(f"    [WARN] the {left_name} key is not unique. Rows sharing a "
              f"barcode will all receive the SAME value.")
    print(f"    -> using '{label}' ({ov:,} matched, {frac:.1%})")
    return lk, rk, label


def attach_stiffness(df):
    """
    Make sure the STIFFNESS_COLS are present on the segmentation table.

    If they already are, nothing is read. Otherwise STIFFNESS_TSV is loaded and
    joined on the barcode. Returns the list of columns actually available.
    """
    have = [c for c in STIFFNESS_COLS if c in df.columns]
    need = [c for c in STIFFNESS_COLS if c not in df.columns]
    if not need:
        print(f"  Stiffness columns already in the metadata: {have}")
        return df, have

    if STIFFNESS_TSV is None or not Path(STIFFNESS_TSV).exists():
        print(f"  WARNING — {need} not in the metadata and STIFFNESS_TSV is "
              f"missing ({STIFFNESS_TSV}). Stiffness layers skipped.")
        return df, have

    print(f"  Reading stiffness from: {Path(STIFFNESS_TSV).name}")
    nano = pd.read_csv(STIFFNESS_TSV, sep="\t", low_memory=False)
    left_key, right_key, key_label = choose_join_key(
        df, nano, "segmentation", Path(STIFFNESS_TSV).name)
    left_name = right_name = key_label

    missing_there = [c for c in need if c not in nano.columns]
    if missing_there:
        print(f"  WARNING — {missing_there} not in "
              f"{Path(STIFFNESS_TSV).name} either. Columns available: "
              f"{[c for c in nano.columns][:30]}")
        need = [c for c in need if c in nano.columns]
        if not need:
            return df, have

    nano = nano.assign(_bc=right_key)[["_bc"] + need].drop_duplicates("_bc")
    n_before = len(df)
    df = df.assign(_bc=left_key).merge(nano, on="_bc", how="left")
    if len(df) != n_before:
        raise ValueError(
            f"Stiffness join changed the row count ({n_before:,} -> "
            f"{len(df):,}); the barcode is not unique in "
            f"{Path(STIFFNESS_TSV).name}.")

    print(f"  Joined on '{left_name}' <- '{right_name}': "
          f"{df['_bc'].isin(set(nano['_bc'])).sum():,} of {len(df):,} bins "
          f"matched")
    for c in need:
        v = pd.to_numeric(df[c], errors="coerce")
        df[c] = v
        print(f"    {c}: {v.notna().sum():,} bins with a value "
              f"({v.notna().mean():.1%}), "
              f"range {np.nanmin(v):.4g} to {np.nanmax(v):.4g}"
              if v.notna().any() else f"    {c}: NO values after the join")
    return df.drop(columns=["_bc"]), have + need


# =============================================================================
# GENE SELECTION FROM THE COMPARTMENT DEG TABLES
# =============================================================================

def _pick_col(cols, wanted, extra_first=None):
    """First matching column name, case-insensitively, else None."""
    lower = {str(c).strip().lower(): c for c in cols}
    for cand in ([extra_first] if extra_first else []) + \
                DEG_COL_CANDIDATES[wanted]:
        if cand is None:
            continue
        hit = lower.get(str(cand).strip().lower())
        if hit is not None:
            return hit
    return None


# A list rather than a bool so _read_deg can set it without a global statement.
_SHRINK_NOTE_SHOWN = []


def _read_deg(path, comp):
    """
    One DEG table, reduced to gene / lfc / padj.

    The gene column is often R's unnamed rowname column, so if no named
    candidate matches, the first column is used and that is stated in the log
    rather than assumed quietly.
    """
    d = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    d = d.loc[:, [c for c in d.columns
                  if not (str(c).startswith("Unnamed:") and d[c].isna().all())]]

    g_col = _pick_col(d.columns, "gene", extra_first=DEG_GENE_COLUMN)
    if g_col is None and len(d.columns):
        g_col = d.columns[0]
        note = f" (no named gene column; using the first, '{g_col}')"
    else:
        note = ""
    l_col = _pick_col(d.columns, "lfc", extra_first=DEG_LFC_COLUMN)
    p_col = _pick_col(d.columns, "padj", extra_first=DEG_PADJ_COLUMN)
    used_raw_p = False
    if p_col is None:
        p_col = _pick_col(d.columns, "pval")
        used_raw_p = p_col is not None

    print(f"    {comp:<3s} {path.name}")
    print(f"        columns -> gene='{g_col}'  log2FC='{l_col}'  "
          f"p='{p_col}'{note}")
    if l_col is None or p_col is None or g_col is None:
        print(f"        [SKIP] could not identify the needed columns. "
              f"Available: {list(d.columns)[:15]}")
        return None
    if used_raw_p:
        print(f"        [WARNING] no adjusted-p column found, so '{p_col}' is "
              f"a RAW p value. DEG_PADJ_MAX is being applied to it unadjusted, "
              f"which is more permissive than it looks.")

    # Say which fold-change estimate was taken when the table offers both.
    # Explained in full the first time, then noted briefly, so four identical
    # paragraphs do not bury the counts underneath them.
    lower = {str(c).strip().lower() for c in d.columns}
    for shrunk, mle in DEG_SHRUNK_MLE_PAIRS:
        if shrunk.lower() in lower and mle.lower() in lower:
            chosen = ("shrunken" if str(l_col).strip().lower() == shrunk.lower()
                      else "unshrunken MLE"
                      if str(l_col).strip().lower() == mle.lower() else "other")
            if not _SHRINK_NOTE_SHOWN:
                print(f"        both '{shrunk}' and '{mle}' are present; "
                      f"using '{l_col}' ({chosen}).")
                print(f"        Shrinkage pulls the inflated fold changes of "
                      f"low-count, high-variance genes back toward zero, so "
                      f"ranking on the shrunken estimate keeps noisy "
                      f"low-expression genes out of the top of the list. "
                      f"Set DEG_LFC_COLUMN to override.")
                _SHRINK_NOTE_SHOWN.append(True)
            else:
                print(f"        ('{l_col}', {chosen}, as above)")
            break

    out = pd.DataFrame({
        "gene": d[g_col].astype(str).str.strip(),
        "lfc":  pd.to_numeric(d[l_col], errors="coerce"),
        "padj": pd.to_numeric(d[p_col], errors="coerce"),
    }).dropna(subset=["gene", "lfc", "padj"])
    out = out[out["gene"] != ""]
    return out


def _load_mechanomics():
    if not MECHANOMICS_CSV.exists():
        print(f"  [WARNING] Mechanomics file not found:\n    {MECHANOMICS_CSV}")
        print("  Every DEG will be treated as a match, so the counts below are "
              "DEG counts, not Mechanomics overlaps.")
        return None
    m = pd.read_csv(MECHANOMICS_CSV, encoding="utf-8-sig")
    col = _pick_col(m.columns, "gene") or m.columns[0]
    n_all = len(m)

    dropped = 0
    if DEG_REQUIRE_IN_MATRIX and "in_expression_matrix" in m.columns:
        keep = m["in_expression_matrix"].astype(str).str.strip().str.lower() \
                .isin(["true", "1", "yes"])
        dropped = int((~keep).sum())
        m = m[keep]
    elif DEG_REQUIRE_IN_MATRIX:
        print("  [NOTE] DEG_REQUIRE_IN_MATRIX is on but the file has no "
              "'in_expression_matrix' column, so no filtering was applied.")

    genes = set(m[col].astype(str).str.strip()) - {"", "nan"}
    print(f"  Mechanomics list: {len(genes):,} genes "
          f"(column '{col}' of {MECHANOMICS_CSV.name})")
    if dropped:
        print(f"    {n_all:,} in the file, {dropped:,} dropped as absent from "
              f"the expression matrix")
    if "source" in m.columns:
        counts = m["source"].value_counts()
        print("    source: " + ", ".join(f"{k}={v:,}" for k, v in
                                         counts.items()))
    return genes


def _ask_int(prompt, default):
    """One integer from the terminal, with the default on Enter."""
    if not DEG_INTERACTIVE:
        print(f"  {prompt} -> {default}  (DEG_INTERACTIVE = False)")
        return default
    if not _sys.stdin or not _sys.stdin.isatty():
        print(f"  {prompt} -> {default}  "
              f"(no interactive terminal; using the default)")
        return default
    while True:
        try:
            raw = input(f"  {prompt} [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  input cancelled; using {default}")
            return default
        if raw == "":
            print(f"    -> {default}")      # echoed so the run log records it
            return default
        try:
            v = int(raw)
            if v < 0:
                print("    please give 0 or more")
                continue
            print(f"    -> {v}")
            return v
        except ValueError:
            print(f"    '{raw}' is not a whole number, try again")


def _rank_deg(d, direction):
    """
    Order one direction's hits best-first.

    One function so the list printed on screen and the genes actually selected
    can never fall out of step. Under DEG_RANK_BY = "lfc" that means largest
    positive log2FC first for up, most negative first for down. Under "padj" it
    is smallest adjusted p first in both directions.
    """
    if DEG_RANK_BY == "lfc":
        return d.sort_values("lfc", ascending=(direction == "down"))
    return d.sort_values("padj", ascending=True)


def deg_gene_selection():
    """
    Build GENES_TO_PLOT / GENE_COMPARTMENTS from the compartment DEG tables.

    Returns (genes, gene_compartments) or (None, None) if nothing usable was
    found, in which case the caller keeps the hand-written lists.
    """
    print("=" * 70)
    print("GENE SELECTION FROM COMPARTMENT DEGs (Aged vs Young)")
    print("=" * 70)
    print(f"  DEG folder: {DEG_DIR}")
    print(f"  Filters   : padj < {DEG_PADJ_MAX}"
          + (f" and (log2FC > {DEG_LFC_MIN} or log2FC < -{DEG_LFC_MIN})"
             if DEG_LFC_MIN else "")
          + f", ranked by {DEG_RANK_BY}")

    mech = _load_mechanomics()
    print()

    tables = {}
    for comp, name in DEG_COMPARTMENT_FILES.items():
        f = DEG_DIR / DEG_FILE_PATTERN.format(name=name)
        if not f.exists():
            print(f"    {comp:<3s} [MISSING] {f.name}")
            continue
        t = _read_deg(f, comp)
        if t is not None:
            tables[comp] = t
    print()

    if not tables:
        print("  No DEG tables could be read. Falling back to the "
              "hand-written GENES_TO_PLOT.\n")
        return None, None

    # ---- counts ----------------------------------------------------------
    picks = {}
    print(f"  {'comp':<5s}{'tested':>9s}{'signif':>9s}{'up':>7s}{'down':>7s}"
          f"{'| mech up':>11s}{'mech down':>11s}")
    for comp, t in tables.items():
        sig = t[t["padj"] < DEG_PADJ_MAX]
        if DEG_LFC_MIN:
            sig = sig[sig["lfc"].abs() > DEG_LFC_MIN]   # strict, see the config
        up, dn = sig[sig["lfc"] > 0], sig[sig["lfc"] < 0]
        m_up = up if mech is None else up[up["gene"].isin(mech)]
        m_dn = dn if mech is None else dn[dn["gene"].isin(mech)]
        picks[comp] = (m_up, m_dn)
        print(f"  {comp:<5s}{len(t):>9,}{len(sig):>9,}{len(up):>7,}"
              f"{len(dn):>7,}{len(m_up):>11,}{len(m_dn):>11,}")
    print()

    # ---- the matching genes themselves -----------------------------------
    rows = []
    for comp, (m_up, m_dn) in picks.items():
        print(f"  {comp}  Mechanomics x DEG matches")
        for label, d, direction in (("UP in aged", m_up, "up"),
                                    ("DOWN in aged", m_dn, "down")):
            if d.empty:
                print(f"      {label:<13s} none")
                continue
            d = _rank_deg(d, direction)
            shown = d.head(DEG_LIST_MAX)
            print(f"      {label:<13s} {len(d)} gene(s); "
                  f"showing {len(shown)}")
            for _, r in shown.iterrows():
                print(f"          {r['gene']:<16s} log2FC={r['lfc']:+.3f}  "
                      f"padj={r['padj']:.3g}")
            if len(d) > len(shown):
                print(f"          ... {len(d) - len(shown)} more "
                      f"(full table written to CSV)")
            for rank, (_, r) in enumerate(d.iterrows(), start=1):
                rows.append({"compartment": comp, "direction": direction,
                             "rank": rank, "gene": r["gene"],
                             "log2FC": r["lfc"], "padj": r["padj"]})
        print()

    if rows:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = OUTPUT_DIR / "deg_mechanomics_matches.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"  Full match table: {out_csv}\n")

    # ---- how many to plot ------------------------------------------------
    n_up = _ask_int("Top UP genes per compartment to plot", DEG_TOP_UP)
    n_dn = _ask_int("Top DOWN genes per compartment to plot", DEG_TOP_DOWN)
    print()

    genes, comps = [], {}
    for comp, (m_up, m_dn) in picks.items():
        chosen = []
        for direction, d, n in (("up", m_up, n_up), ("down", m_dn, n_dn)):
            if d.empty or n <= 0:
                continue
            chosen += _rank_deg(d, direction).head(n)["gene"].tolist()
        if not chosen:
            print(f"  {comp}: nothing selected")
            continue
        print(f"  {comp}: {len(chosen)} gene(s) -> {chosen}")
        for g in chosen:
            if g not in genes:
                genes.append(g)
            comps.setdefault(g, [])
            if comp not in comps[g]:
                comps[g].append(comp)

    if not genes:
        print("\n  Nothing selected. Falling back to the hand-written "
              "GENES_TO_PLOT.\n")
        return None, None

    multi = {g: c for g, c in comps.items() if len(c) > 1}
    if multi:
        print(f"\n  In more than one compartment, so drawn in each: {multi}")
    print(f"\n  {len(genes)} unique gene(s) selected. Each is drawn per "
          f"compartment and per numbered F / CL feature, so expect roughly "
          f"{len(genes) * 4} gene files per tissue.")
    print("=" * 70 + "\n")
    return genes, comps


def report_handpicked_deg_status(genes, mech_genes=None):
    """
    For each hand-picked gene, in each compartment: did it make the DEG list,
    and if not, what were its actual numbers?

    Reads the same four DEG tables the automatic selection uses, so "would have
    made the list" here means exactly what it means there: padj < DEG_PADJ_MAX
    and |log2FC| > DEG_LFC_MIN, and, for the automatic list, also being on the
    Mechanomics set. A gene can fail on any one of those and the reason is
    printed rather than left to be inferred.
    """
    if not genes:
        return
    print("\n" + "=" * 70)
    print("HAND-PICKED GENES: DEG STATUS")
    print("=" * 70)
    print(f"  DEG gate: padj < {DEG_PADJ_MAX} and |log2FC| > {DEG_LFC_MIN}")

    rows = []
    for comp, name in DEG_COMPARTMENT_FILES.items():
        f = DEG_DIR / DEG_FILE_PATTERN.format(name=name)
        if not f.exists():
            print(f"  [MISSING] {f.name}; {comp} cannot be reported")
            continue
        t = _read_deg(f, comp)
        if t is None:
            continue
        idx = t.set_index("gene")
        for g in genes:
            if g not in idx.index:
                rows.append({"gene": g, "compartment": comp,
                             "in_deg_table": False, "log2FC": np.nan,
                             "padj": np.nan, "passes_padj": False,
                             "passes_lfc": False, "is_DEG": False,
                             "on_mechanomics_list": (
                                 g in mech_genes if mech_genes else np.nan),
                             "note": "not present in the DEG table"})
                continue
            r = idx.loc[g]
            if isinstance(r, pd.DataFrame):     # duplicate symbol
                r = r.iloc[0]
            lfc, padj = float(r["lfc"]), float(r["padj"])
            p_ok = padj < DEG_PADJ_MAX
            l_ok = abs(lfc) > DEG_LFC_MIN
            why = []
            if not p_ok:
                why.append(f"padj {padj:.3g} not < {DEG_PADJ_MAX}")
            if not l_ok:
                why.append(f"|log2FC| {abs(lfc):.3f} not > {DEG_LFC_MIN}")
            rows.append({"gene": g, "compartment": comp, "in_deg_table": True,
                         "log2FC": lfc, "padj": padj,
                         "passes_padj": p_ok, "passes_lfc": l_ok,
                         "is_DEG": bool(p_ok and l_ok),
                         "on_mechanomics_list": (
                             g in mech_genes if mech_genes else np.nan),
                         "note": "DEG" if (p_ok and l_ok) else "; ".join(why)})

    if not rows:
        print("  nothing could be read; check DEG_DIR")
        return
    tab = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / HANDPICKED_DIR / "handpicked_deg_status.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False)

    print()
    for g in genes:
        sub = tab[tab["gene"] == g]
        hits = sub[sub["is_DEG"]]["compartment"].tolist()
        mech = sub["on_mechanomics_list"].iloc[0] if len(sub) else np.nan
        mech_txt = ("" if not isinstance(mech, (bool, np.bool_))
                    else ("  [on the Mechanomics list]" if mech
                          else "  [NOT on the Mechanomics list]"))
        if hits:
            print(f"  {g}: DEG in {hits}{mech_txt}")
        else:
            print(f"  {g}: NOT a DEG in any compartment{mech_txt}")
        for _, r in sub.iterrows():
            if not r["in_deg_table"]:
                print(f"      {r['compartment']:<3s} not in the DEG table")
            else:
                flag = "DEG" if r["is_DEG"] else "   "
                print(f"      {r['compartment']:<3s} {flag}  "
                      f"log2FC = {r['log2FC']:+.3f}   padj = {r['padj']:.3g}"
                      + ("" if r["is_DEG"] else f"   ({r['note']})"))
    print(f"\n  Written: {out}")
    print("  Note: a gene can be a real DEG here and still be absent from the "
          "automatic plotting list, because that list is additionally "
          "intersected with the Mechanomics set and capped at the top N per "
          "direction.")


def attach_gene_expression(df, genes):
    """
    Pull the requested genes out of the step9 expression matrix and join them
    onto the segmentation table by barcode.

    The matrix rows line up with EXPR_META_FILE, not with the segmentation
    table, so the join goes through that file's barcode column rather than
    assuming row order.
    """
    if not genes:
        return df, []

    try:
        from scipy import io as sio
    except Exception as e:
        print(f"  WARNING — scipy.io unavailable ({e}); gene layers skipped.")
        return df, []

    mtx_path = Path(EXPR_DIR) / EXPR_MATRIX_FILE
    if not mtx_path.exists():
        print(f"  WARNING — expression matrix not found at {mtx_path}. "
              f"Gene layers skipped.")
        return df, []

    with open(Path(EXPR_DIR) / EXPR_FEATURES_FILE, encoding="utf-8") as f:
        feats = [l.strip() for l in f]
    with open(Path(EXPR_DIR) / EXPR_BARCODES_FILE, encoding="utf-8") as f:
        n_bc = sum(1 for _ in f)

    lookup = {g.lower(): g for g in feats}
    found, absent = [], []
    for g in genes:
        if g in feats:
            found.append(g)
        elif g.lower() in lookup:
            real = lookup[g.lower()]
            print(f"  NOTE — '{g}' matched '{real}' by case-insensitive lookup.")
            found.append(real)
        else:
            absent.append(g)
    if absent:
        for g in absent:
            near = [f for f in feats
                    if f.lower().startswith(g.lower()[:3])][:8]
            print(f"  WARNING — gene '{g}' is not in the expression matrix. "
                  f"Similar symbols: {near}")
    if not found:
        return df, []

    print(f"  Loading expression matrix for {len(found)} gene(s): {found}")
    X = sio.mmread(mtx_path).tocsr()
    if X.shape[0] == len(feats) and X.shape[1] == n_bc:
        X = X.T.tocsr()
    if X.shape[1] != len(feats):
        print(f"  WARNING — matrix shape {X.shape} does not match "
              f"{len(feats)} features / {n_bc} barcodes. Gene layers skipped.")
        return df, []
    X = X.tocsc()

    emeta = pd.read_csv(Path(EXPR_DIR) / EXPR_META_FILE, sep="\t",
                        low_memory=False)
    if len(emeta) != X.shape[0]:
        print(f"  WARNING — {EXPR_META_FILE} has {len(emeta):,} rows but the "
              f"matrix has {X.shape[0]:,}. Gene layers skipped.")
        return df, []

    left_key, right_key, key_label = choose_join_key(
        df, emeta, "segmentation", EXPR_META_FILE)
    left_name = right_name = key_label

    idx = {g: i for i, g in enumerate(feats)}
    # Built as one dict then framed once. Assigning ~200 columns one at a time
    # to a multi-million-row frame is what triggers pandas' "highly fragmented"
    # warning, and it is genuinely slow, not just noisy.
    cols = {"_bc": right_key.values}
    for g in found:
        cols[g] = np.asarray(X[:, idx[g]].todense()).ravel()
    expr = pd.DataFrame(cols).drop_duplicates("_bc")

    # A gene already present on df would make merge rename BOTH sides to
    # <gene>_x / <gene>_y, and the lookup below would then fail with a KeyError
    # that says nothing about the real cause. Dropping first makes a repeat
    # load idempotent instead.
    clash = [g for g in found if g in df.columns]
    if clash:
        print(f"  {len(clash)} gene(s) already loaded; refreshing rather than "
              f"creating duplicate columns: {clash[:10]}"
              f"{' ...' if len(clash) > 10 else ''}")
        df = df.drop(columns=clash)

    n_before = len(df)
    df = df.assign(_bc=left_key).merge(expr, on="_bc", how="left")
    if len(df) != n_before:
        raise ValueError(
            f"Expression join changed the row count ({n_before:,} -> "
            f"{len(df):,}); the barcode is not unique in {EXPR_META_FILE}.")
    n_match = int(df["_bc"].isin(set(expr["_bc"])).sum())
    print(f"  Joined on '{left_name}' <- '{right_name}': "
          f"{n_match:,} of {len(df):,} bins matched")
    for g in found:
        v = df[g]
        print(f"    {g}: {v.notna().sum():,} bins, "
              f"{(v.fillna(0) > 0).sum():,} non-zero, "
              f"max {np.nanmax(v) if v.notna().any() else float('nan'):.3f}")
    return df.drop(columns=["_bc"]), found


# =============================================================================
# VALUE LAYERS: DRAWING
# =============================================================================

def make_cmap(spec, name="value"):
    """A matplotlib colormap name, or a list of hex stops to interpolate."""
    if isinstance(spec, str):
        return plt.get_cmap(spec)
    return mcolors.LinearSegmentedColormap.from_list(name, list(spec))


def value_limits(values):
    """Percentile-clipped vmin / vmax, ignoring NaNs."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if v.empty:
        return None, None
    lo, hi = np.nanpercentile(v, VALUE_CLIP_PCT)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        hi = lo + 1e-9
    return float(lo), float(hi)


def scatter_val(ax, x, y, values, cmap, vmin, vmax, s):
    """
    Square-marker scatter coloured by a continuous value.

    Bins with no value are drawn first in MISSING_VALUE_COLOUR so the feature
    keeps its shape, otherwise a follicle only half-covered by indentation
    points would look like half a follicle.
    """
    v = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), float)
    ok = np.isfinite(v)
    if DRAW_MISSING_BINS and (~ok).any():
        scatter_sq(ax, np.asarray(x)[~ok], np.asarray(y)[~ok],
                   MISSING_VALUE_COLOUR, s, zorder=2)
    if ok.any():
        sc = ax.scatter(np.asarray(x)[ok], np.asarray(y)[ok], c=v[ok],
                        cmap=cmap, vmin=vmin, vmax=vmax, s=s, marker="s",
                        linewidths=0, zorder=3, rasterized=RASTERIZE_DOTS)
        if not hasattr(ax, "_bin_collections"):
            ax._bin_collections = []
        ax._bin_collections.append(sc)
    return int(ok.sum()), int((~ok).sum())


def save_colourbar(cmap, vmin, vmax, label, stem):
    """
    One small standalone colour bar per value, so the layers stay clean.

    Written to OUTPUT_DIR itself, not inside Young/ or Aged/. Under
    VALUE_SCALE_SCOPE = "global" there is exactly one scale per value shared by
    both ages, so one bar in the parent is the honest place for it. Under
    "tissue" the per-tissue bars also land here, with the tissue ID in the file
    name, so a bar can never sit next to layers it does not describe.
    """
    if not SAVE_COLOURBARS:
        return
    fig, ax = plt.subplots(figsize=(1.2, 3.2))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    tr = COLOURBAR_TRANSPARENT
    fc = "none" if tr else "white"
    written = []
    for ext, dpi in (("png", PNG_DPI), ("tiff", TIFF_DPI), ("svg", None)):
        if ext not in EXPORT_FORMATS:
            continue
        kw = {"dpi": dpi} if dpi else {}
        fig.savefig(OUTPUT_DIR / f"{stem}.{ext}", bbox_inches="tight",
                    facecolor=fc, edgecolor="none", transparent=tr, **kw)
        written.append("." + ext)
    plt.close(fig)
    print(f"    saved: {stem}  ({' / '.join(written)})  [colour bar]")


def plot_value_layers(tissue_df, tissue_id, value_col, suffix, cmap,
                      vmin, vmax, xlim, ylim, s, comps_present,
                      kind="stiffness"):
    """
    The same crops as the compartment layers, coloured by value_col.

    <ID>_<comp>only_<suffix>   whole compartment
    <ID>-<comp><n>_<suffix>    each numbered F / CL feature

    Canvas and axis limits are identical to every other layer, so these drop
    straight onto the compartment layers in Illustrator.
    """
    if value_col not in tissue_df.columns:
        return

    # For gene layers both levels land in that gene's own folder, so a single
    # Genes/<gene>/ directory holds everything about that gene.
    gene_key = suffix if kind in ("gene", "handpicked") else None
    comp_dir = out_dir_for(tissue_id, kind, "compartment", gene=gene_key)
    feat_dir = out_dir_for(tissue_id, kind, "feature", gene=gene_key)
    fmts = EXPORT_FORMATS_BY_KIND.get(kind)
    mode = grey_mode_for(kind)
    bg_roi  = None if mode == "none" else grey_bg_frame(tissue_df)
    bg_feat = bg_roi if GREY_BG_ON_FEATURE_LAYERS else None

    for comp in ("F", "CL", "S", "V"):
        if comp not in comps_present:
            continue
        # Hand-picked genes deliberately ignore GENE_COMPARTMENTS: the reason
        # for naming a gene by hand is usually to look where the DEG test did
        # not point you.
        if kind == "handpicked":
            allowed = HANDPICKED_COMPARTMENTS
        else:
            allowed = (GENE_COMPARTMENTS or {}).get(suffix)
        if allowed is not None and comp not in allowed:
            continue
        mask = tissue_df[COL_SEG].apply(lambda v: parse_seg(v)[1] == comp)
        sub = tissue_df[mask]
        if sub.empty:
            continue

        # Decided before drawing, so the skip does not depend on having built a
        # figure first. emit_layer may draw the same content more than once.
        n_val = int(np.isfinite(pd.to_numeric(
            pd.Series(sub[value_col].values), errors="coerce")).sum())
        if n_val == 0:
            print(f"    skipped: {tissue_id}_{comp}only_{suffix}  "
                  f"(no bins with a value)")
        else:
            emit_layer(
                lambda ax, _sub=sub: scatter_val(
                    ax, _sub[COL_X].values, _sub[COL_Y].values,
                    _sub[value_col].values, cmap, vmin, vmax, s),
                f"{tissue_id}_{comp}only_{suffix}", comp_dir,
                xlim, ylim, f"{tissue_id}  ·  {comp}  ·  {suffix}",
                bg_roi, s, formats=fmts, mode=mode)

        if comp not in ("F", "CL"):
            continue

        nums = sorted(sub[COL_SEG].apply(lambda v: parse_seg(v)[2])
                      .dropna().unique())
        for num in nums:
            rows = sub[sub[COL_SEG].apply(
                lambda v: parse_seg(v)[2]) == num]
            if rows.empty:
                continue
            fig, ax = new_layer_fig(xlim, ylim, s, bg_df=bg_feat)
            n_val, n_na = scatter_val(ax, rows[COL_X].values,
                                      rows[COL_Y].values,
                                      rows[value_col].values,
                                      cmap, vmin, vmax, s)
            apply_style_bare(
                ax, xlim, ylim,
                f"{tissue_id}  ·  {comp}{int(num)}  ·  {suffix}")
            if n_val == 0:
                plt.close(fig)
                print(f"    skipped: {tissue_id}-{comp}{int(num)}_{suffix}  "
                      f"(no bins with a value; {n_na} bins have none)")
            else:
                save_png_svg(fig, f"{tissue_id}-{comp}{int(num)}_{suffix}",
                             out_dir=feat_dir, formats=fmts)


# =============================================================================
# PARALLEL EXECUTION
#
# Each job is one call to one of the plotting functions above. The frame a job
# needs is not pickled into the job: it is written once to a temp file by the
# parent and loaded lazily by whichever worker touches it first, then kept in
# that worker's cache. Pickling a 30,000-row frame into every one of several
# hundred jobs would cost more than the rendering it is meant to speed up.
# =============================================================================

_FRAME_CACHE = {}      # per worker: frame key -> DataFrame
_FRAME_DIR = None      # per worker: where those frames live

SCRATCH_PREFIX = "plotseg_frames_"


def _dir_size_mb(d):
    try:
        return sum(f.stat().st_size for f in Path(d).rglob("*")
                   if f.is_file()) / 1e6
    except Exception:
        return float("nan")


def remove_scratch(d):
    """
    Delete a scratch directory and SAY whether it worked.

    The previous version passed ignore_errors=True, so a failure vanished and
    the only evidence was a stray plotseg_frames_* folder with no explanation.
    On Windows a worker process or an indexer can still hold a handle for a
    moment after the pool closes, so one retry is worth it; if it still fails
    the path is printed rather than swallowed.
    """
    d = Path(d)
    if not d.exists():
        return True
    mb = _dir_size_mb(d)
    for attempt in (1, 2):
        try:
            shutil.rmtree(d)
            print(f"  scratch removed: {d.name}  ({mb:,.0f} MB freed)")
            return True
        except Exception as e:
            if attempt == 1:
                time.sleep(1.0)     # let a lingering handle close
                continue
            print(f"  [WARNING] could not delete the scratch folder:\n"
                  f"    {d}\n"
                  f"    {type(e).__name__}: {e}\n"
                  f"    It holds {mb:,.0f} MB of temporary pickled frames and "
                  f"nothing you need. Safe to delete by hand.")
    return False


def sweep_stale_scratch(roots):
    """Remove plotseg_frames_* left behind by an interrupted earlier run."""
    seen = set()
    for root in roots:
        root = Path(root)
        if not root.exists() or root in seen:
            continue
        seen.add(root)
        try:
            stale = [p for p in root.iterdir()
                     if p.is_dir() and p.name.startswith(SCRATCH_PREFIX)]
        except Exception:
            continue
        if not stale:
            continue
        total = sum(_dir_size_mb(p) for p in stale)
        print(f"\n[SCRATCH] {len(stale)} leftover folder(s) in {root} "
              f"({total:,.0f} MB) from a run that did not finish cleanly:")
        for p in stale:
            print(f"    {p.name}")
            remove_scratch(p)


def _worker_init(frame_dir, shade_maps):
    """
    Runs once per worker process.

    SHADE_MAPS has to be handed over explicitly. On Linux the fork start method
    would inherit it, but Windows uses spawn: the worker re-imports this module
    from scratch, where SHADE_MAPS is an empty dict, and every feature would
    come out in the wrong shade. Passing it makes the two platforms behave the
    same.
    """
    global _FRAME_DIR
    _FRAME_DIR = frame_dir
    SHADE_MAPS.clear()
    SHADE_MAPS.update(shade_maps)
    matplotlib.use("Agg", force=True)


def _frame(key):
    """
    The frame for this job.

    Only ONE frame is kept per worker. The cache used to grow without bound,
    so a worker that touched several tissues held all of them at once and 60
    workers ran the machine out of memory.
    """
    if key not in _FRAME_CACHE:
        _FRAME_CACHE.clear()
        _FRAME_CACHE[key] = pd.read_pickle(Path(_FRAME_DIR) / f"{key}.pkl")
    return _FRAME_CACHE[key]


def _run_job(job):
    """
    Execute one plotting job and return (label, captured output).

    Output is captured rather than printed, because several workers printing at
    once would interleave into an unreadable run log. The parent replays it in
    job order. A job that raises returns its traceback instead of killing the
    pool, so one bad feature cannot lose the whole run.
    """
    buf = io.StringIO()
    kind = job["kind"]
    with redirect_stdout(buf):
        try:
            d = _frame(job["frame"])
            a = job["args"]
            if kind == "single_comp":
                plot_single_compartment(d, a["tid"], a["comp"],
                                        a["xlim"], a["ylim"], a["s"])
            elif kind == "combined":
                plot_tissue_combined(d, a["tid"], a["xlim"], a["ylim"], a["s"])
            elif kind == "flat":
                plot_tissue_flat(d, a["tid"], a["xlim"], a["ylim"], a["s"])
            elif kind == "comp_only":
                plot_compartment_only(d, a["tid"], a["comp"],
                                      a["xlim"], a["ylim"], a["s"])
            elif kind == "features":
                plot_individual_features(d, a["tid"], a["comp"],
                                         a["xlim"], a["ylim"], a["s"])
            elif kind == "values":
                plot_value_layers(d, a["tid"], a["col"], a["suffix"],
                                  a["cmap"], a["lo"], a["hi"],
                                  a["xlim"], a["ylim"], a["s"],
                                  a["comps"], kind=a["vkind"])
            elif kind == "slice":
                plot_tissue_slice(d, a["sid"], span=a.get("span"))
            else:
                print(f"    [BUG] unknown job kind {kind!r}")
        except Exception:
            print(f"    [FAILED] {job['label']}")
            traceback.print_exc(file=_sys.stdout)
    return job["label"], buf.getvalue()


def available_memory_bytes():
    """Free RAM, or None when it cannot be determined on this platform."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return None


def resolve_workers(n_jobs, frame_bytes=0):
    """
    How many processes to actually start, and why.

    Capped by memory as well as by cores. Each worker holds one frame, so the
    ceiling is available RAM x WORKER_MEMORY_FRACTION divided by the largest
    frame. Without this the run dies with MemoryError partway through and the
    failed jobs are silently missing from the output.
    """
    cores = os.cpu_count() or 1
    want = cores if N_WORKERS is None else int(N_WORKERS)
    n = max(1, min(want, cores, n_jobs))
    if n_jobs < PARALLEL_MIN_JOBS:
        n = 1
    print(f"  requested {N_WORKERS}, machine has {cores} core(s), "
          f"{n_jobs} job(s) -> using {n} worker(s)")
    if want > cores:
        print(f"  [NOTE] {want} was capped to {cores}; more processes than "
              f"cores would only add context switching")

    if frame_bytes > 0:
        avail = available_memory_bytes()
        per = frame_bytes * 3          # pickle load peaks well above the frame
        print(f"  largest frame {frame_bytes / 2**20:,.0f} MB, "
              f"budget {per / 2**20:,.0f} MB per worker")
        if avail is None:
            print("  [NOTE] free memory unknown (pip install psutil to enable "
                  "the memory cap); worker count not adjusted")
        else:
            cap = max(1, int(avail * WORKER_MEMORY_FRACTION // per))
            print(f"  {avail / 2**30:,.1f} GiB free x "
                  f"{WORKER_MEMORY_FRACTION:.0%} -> room for {cap} worker(s)")
            if cap < n:
                print(f"  [CAP] {n} -> {cap} worker(s) to stay inside memory")
                n = cap
    return n


def execute_jobs(jobs, frame_dir, frame_bytes=0):
    """
    Run the jobs, in parallel when that is worth it, serially otherwise.

    Falls back to serial if the pool cannot start at all, rather than failing
    the run, and says so.
    """
    if not jobs:
        return
    n = resolve_workers(len(jobs), frame_bytes=frame_bytes)
    t0 = time.perf_counter()

    if n <= 1:
        _worker_init(frame_dir, dict(SHADE_MAPS))
        for job in jobs:
            _, out = _run_job(job)
            if out:
                print(out, end="")
    else:
        try:
            from concurrent.futures import ProcessPoolExecutor
            ctx = mp.get_context("spawn")     # same behaviour on Windows/Linux
            with ProcessPoolExecutor(
                    max_workers=n, mp_context=ctx,
                    initializer=_worker_init,
                    initargs=(str(frame_dir), dict(SHADE_MAPS))) as ex:
                # Replayed in submission order so the log reads like a serial
                # run even though the work did not happen in that order.
                for _, out in ex.map(_run_job, jobs):
                    if out:
                        print(out, end="")
        except Exception as e:
            print(f"  [WARNING] parallel execution failed ({type(e).__name__}: "
                  f"{e}); falling back to serial. Output is unaffected, only "
                  f"the wall-clock.")
            _worker_init(frame_dir, dict(SHADE_MAPS))
            for job in jobs:
                _, out = _run_job(job)
                if out:
                    print(out, end="")

    dt = time.perf_counter() - t0
    print(f"\n  {len(jobs)} job(s) in {dt:,.1f} s on {n} worker(s) "
          f"= {len(jobs) / max(dt, 1e-9):.2f} jobs/s")
    if n > 1:
        print(f"  If this is far below {n}x the serial rate, the limit is the "
              f"disk rather than the CPU, and more workers will not help.")


# =============================================================================
# PER-TISSUE-SLICE COMPARTMENT MAP
# =============================================================================

def plot_tissue_slice(slice_df, slice_id, span=None):
    """
    One plot per Tissue-slices value, coloured by Compartments column.
    Unassigned bins are omitted (no shadow — this is a clean compartment map).

    span = (width_um, height_um) draws every slice at the same micrometres per
    inch, so the exported files can be placed side by side without rescaling.
    """
    assigned = slice_df[slice_df[COL_COMPARTMENT].isin(BASE_COLOURS)].copy()
    if assigned.empty:
        return

    xs = assigned[COL_X]
    ys = assigned[COL_Y]
    s = marker_s(0, 0)

    if span is not None:
        w, h = span
        cx = (xs.min() + xs.max()) / 2.0
        cy = (ys.min() + ys.max()) / 2.0
        xlim = (cx - w / 2.0, cx + w / 2.0)
        ylim = (cy - h / 2.0, cy + h / 2.0)
    else:
        x_range = xs.max() - xs.min()
        y_range = ys.max() - ys.min()
        pad_x = x_range * 0.04
        pad_y = y_range * 0.04
        xlim = (xs.min() - pad_x, xs.max() + pad_x)
        ylim = (ys.min() - pad_y, ys.max() + pad_y)

    fig, ax = new_fig()
    handles = []

    for comp, base in BASE_COLOURS.items():
        sub = assigned[assigned[COL_COMPARTMENT] == comp]
        if sub.empty:
            continue
        scatter_sq(ax, sub[COL_X].values, sub[COL_Y].values, base, s)
        handles.append(patch(base, comp))

    apply_style(ax, xlim, ylim,
                f"{slice_id}  ·  Compartments", handles)
    save_all(fig, f"{slice_id}-Compartments",
             out_dir=out_dir_for(slice_id, "overview"))


# =============================================================================
# ALL-CAPTURE-AREAS COMBINED MAP
# =============================================================================

def plot_all_capture_areas(df):
    assigned = df[df[COL_COMPARTMENT].isin(BASE_COLOURS)].copy()
    if assigned.empty:
        print("  No assigned Compartments rows — skipping all-CA plot.")
        return

    xs = assigned[COL_X]
    ys = assigned[COL_Y]
    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()
    s = marker_s(x_range, y_range)

    pad_x = x_range * 0.02
    pad_y = y_range * 0.02
    xlim = (xs.min() - pad_x, xs.max() + pad_x)
    ylim = (ys.min() - pad_y, ys.max() + pad_y)

    # Use a wider canvas for the combined overview
    fig, ax = plt.subplots(figsize=(20, 14))
    handles = []

    for comp, base in BASE_COLOURS.items():
        sub = assigned[assigned[COL_COMPARTMENT] == comp]
        if sub.empty:
            continue
        scatter_sq(ax, sub[COL_X].values, sub[COL_Y].values, base, s)
        handles.append(patch(base, comp))

    apply_style(ax, xlim, ylim,
                "All Capture Areas  ·  Tissue Compartments", handles)
    # Young and aged tissues are both on this one canvas, so it belongs to
    # neither age folder. Stays in the parent alongside the colour bars.
    save_all(fig, "All_CaptureAreas_Compartments")


# =============================================================================
# MAIN
# =============================================================================

def _run():
    # Declared up front: Python requires the global statement to precede any
    # use of the name in the function, and the header below reads them.
    global GENES_TO_PLOT, GENE_COMPARTMENTS

    print("=" * 70)
    print("plot_segmentation run")
    print(f"Date/time     : {datetime.now().isoformat(timespec='seconds')}")
    print(f"Metadata TSV  : {METADATA_TSV}")
    print(f"Stiffness TSV : {STIFFNESS_TSV}")
    print(f"Expression dir: {EXPR_DIR}")
    print(f"Gene source   : "
          + ("compartment DEGs x Mechanomics (chosen below)"
             if DEG_GENE_SELECTION
             else f"hand-written list, {len(GENES_TO_PLOT)} gene(s)"))
    print(f"EXTENT_SCOPE={EXTENT_SCOPE}  SHADE_SCOPE={SHADE_SCOPE}  "
          f"VALUE_SCALE_SCOPE={VALUE_SCALE_SCOPE}")
    print(f"DRAW_MISSING_BINS={DRAW_MISSING_BINS}  "
          f"GENE_ZERO_AS_BLANK={GENE_ZERO_AS_BLANK}")
    print(f"PLOT_FLAT_ROI={PLOT_FLAT_ROI} "
          f"(suffix '{FLAT_ROI_SUFFIX}', halos={FLAT_ROI_HALOS})")
    print(f"OVERVIEW_SHOW_AXES={OVERVIEW_SHOW_AXES}  "
          f"TRANSPARENT_BACKGROUND={TRANSPARENT_BACKGROUND}  "
          f"COLOURBAR_TRANSPARENT={COLOURBAR_TRANSPARENT}")
    print(f"RASTERIZE_DOTS={RASTERIZE_DOTS} at {RASTER_DPI} dpi in the SVG  "
          f"RASTERIZE_HALOS={RASTERIZE_HALOS}")
    print(f"EXPORT_FORMATS={EXPORT_FORMATS}  "
          f"per-kind overrides={EXPORT_FORMATS_BY_KIND}")
    print(f"GENE_GROUP_BY='{GENE_GROUP_BY}' (root '{GENES_ROOT_DIR}')")
    if HANDPICKED_GENES:
        print(f"Hand-picked : {len(HANDPICKED_GENES)} gene(s) "
              f"{HANDPICKED_GENES} -> '{HANDPICKED_DIR}', tissues="
              f"{HANDPICKED_TISSUES or 'all'}, compartments="
              f"{HANDPICKED_COMPARTMENTS or 'all'}")
    print(f"N_WORKERS={N_WORKERS} requested, {os.cpu_count()} core(s) "
          f"available on this machine")
    print(f"GREY_BG per-kind overrides: {GREY_BG_MODE_BY_KIND}")
    print(f"GREY_BG mode={GREY_BG_MODE}  scope={GREY_BG_SCOPE}  "
          f"suffix='{GREY_BG_SUFFIX}'  on features={GREY_BG_ON_FEATURE_LAYERS}")
    print(f"SPLIT_COMPARTMENT_AND_FEATURES={SPLIT_COMPARTMENT_AND_FEATURES}")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}")
    if ORGANISE_OUTPUT:
        print("Layout: ORGANISE_OUTPUT = True")
        print(f"  <parent>/                       colour bars, "
              f"All_CaptureAreas_Compartments, run log")
        if SPLIT_COMPARTMENT_AND_FEATURES:
            plan = [(SUBDIR_OVERVIEW,   "whole tissue, ROI, capture-area maps"),
                    (SUBDIR_COMP,       "<ID>_Fonly / _CLonly / _Sonly / _Vonly"),
                    (SUBDIR_FEAT,       "<ID>-F1, <ID>-CL1, <ID>-F1_halo"),
                    (SUBDIR_STIFF_COMP, "stiffness, whole compartment"),
                    (SUBDIR_STIFF_FEAT, "stiffness, numbered F / CL"),
                    (SUBDIR_GENE_COMP,  "genes, whole compartment"),
                    (SUBDIR_GENE_FEAT,  "genes, numbered F / CL")]
        else:
            plan = [(SUBDIR_OVERVIEW,  "whole tissue, ROI, capture-area maps"),
                    (SUBDIR_FEATURES,  "ST bins + halo, halo-only"),
                    (SUBDIR_STIFFNESS, ", ".join(STIFFNESS_COLS)),
                    (SUBDIR_GENES,     f"{len(GENES_TO_PLOT)} gene layer(s)")]
        for age in (DIR_YOUNG, DIR_AGED):
            for sub, what in plan:
                print(f"  {age}/{sub}/".ljust(44) + what)
        print(f"  {DIR_UNGROUPED}/...                     "
              f"anything whose ID starts with neither Y nor A")
    else:
        print("Layout: ORGANISE_OUTPUT = False, everything flat in the "
              "parent folder")
    print()

    # Done first, before anything slow, so the two prompts are answered up
    # front and the rest of the run needs no attention.
    if DEG_GENE_SELECTION:
        sel_genes, sel_comps = deg_gene_selection()
        if sel_genes:
            GENES_TO_PLOT, GENE_COMPARTMENTS = sel_genes, sel_comps

    if HANDPICKED_GENES and HANDPICKED_REPORT_DEG_STATUS:
        mech = None
        if MECHANOMICS_CSV.exists():
            try:
                _m = pd.read_csv(MECHANOMICS_CSV, encoding="utf-8-sig")
                if "in_expression_matrix" in _m.columns:
                    _m = _m[_m["in_expression_matrix"].astype(str)
                            .str.lower().isin(["true", "1", "yes"])]
                mech = set(_m["gene"].astype(str).str.strip())
            except Exception as e:
                print(f"[NOTE] could not read the Mechanomics set ({e})")
        report_handpicked_deg_status(HANDPICKED_GENES, mech)
    print(f"Genes to plot : {len(GENES_TO_PLOT)} -> {GENES_TO_PLOT}")
    print(f"Gene->compartment restriction: "
          f"{'on' if GENE_COMPARTMENTS else 'off'}\n")

    print(f"Loading: {METADATA_TSV.name}")
    df = pd.read_csv(METADATA_TSV, sep="\t", low_memory=False)
    print(f"  {len(df):,} rows  ×  {df.shape[1]} cols")

    # Column check
    required = [COL_X, COL_Y, COL_SEG, COL_COMPARTMENT,
                COL_OVARY_ID, COL_TISSUE_SLICE]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nERROR — missing columns: {missing}")
        print(f"Available: {df.columns.tolist()}")
        sys.exit(1)

    df[COL_X] = pd.to_numeric(df[COL_X], errors="coerce")
    df[COL_Y] = pd.to_numeric(df[COL_Y], errors="coerce")
    df = df.dropna(subset=[COL_X, COL_Y])
    print(f"  After coordinate coercion: {len(df):,} rows\n")

    # Diagnostic: show unique Compartments values for colour-mapping check
    comp_vals = df[COL_COMPARTMENT].dropna().unique()
    print(f"Unique Compartments values: {sorted(comp_vals)}")
    unmatched = [v for v in comp_vals if v not in BASE_COLOURS]
    if unmatched:
        print(f"  WARNING — not in colour map: {unmatched}")
    print()

    # Diagnostic: show unique ROI-Segmentation labels
    seg_assigned = [v for v in df[COL_SEG].dropna().unique()
                    if parse_seg(v)[1] is not None]
    print(f"Assigned ROI-Segmentation labels ({len(seg_assigned)}):")
    for lab in sorted(seg_assigned):
        print(f"  {lab}")
    print()

    # -------------------------------------------------------------------------
    # One lightness ramp per compartment across every tissue
    # -------------------------------------------------------------------------
    if SHADE_SCOPE == "global":
        print("Feature shade ramps (shared across tissues):")
        build_shade_maps(df)
        print()

    # -------------------------------------------------------------------------
    # First pass: resolve each tissue's bins, then fix one common axis span
    # -------------------------------------------------------------------------
    tissue_frames = {}
    for tissue_id in sorted(df[COL_OVARY_ID].dropna().astype(str).unique()):
        tissue_all = df[df[COL_OVARY_ID].astype(str) == tissue_id].copy()
        seg_ok = tissue_all[COL_SEG].apply(
            lambda v: parse_seg(v)[1] is not None)
        if not seg_ok.any():
            continue
        tissue_df = tissue_all
        roi_slices_kept = None
        if RESTRICT_TO_ROI_SLICES and COL_TISSUE_SLICE in tissue_all.columns:
            slice_str = tissue_all[COL_TISSUE_SLICE].astype(str)
            roi_slices = sorted(set(slice_str[seg_ok.values]))
            dropped = [sl for sl in sorted(set(slice_str))
                       if sl not in roi_slices]
            if roi_slices:
                roi_slices_kept = roi_slices
                tissue_df = tissue_all[slice_str.isin(roi_slices)].copy()
                if dropped:
                    print(f"  {tissue_id}: ROI on {roi_slices}; dropping "
                          f"unannotated slice(s) {dropped} "
                          f"({len(tissue_all) - len(tissue_df):,} bins)")
        tissue_frames[tissue_id] = (tissue_df, roi_slices_kept)

    global_span = None
    if EXTENT_SCOPE == "global" and tissue_frames:
        w = max((t[COL_X].max() - t[COL_X].min())
                for t, _ in tissue_frames.values())
        h = max((t[COL_Y].max() - t[COL_Y].min())
                for t, _ in tissue_frames.values())
        global_span = (w * (1 + 2 * EXTENT_PAD), h * (1 + 2 * EXTENT_PAD))
        print(f"\nCommon axis span for every tissue: "
              f"{global_span[0]:,.0f} x {global_span[1]:,.0f} um "
              f"(largest tissue sets it)")
        for tid, (t, _) in sorted(tissue_frames.items()):
            print(f"    {tid}: {t[COL_X].max() - t[COL_X].min():,.0f} x "
                  f"{t[COL_Y].max() - t[COL_Y].min():,.0f} um")
    s_global = (marker_s(*global_span) if global_span else None)
    print("  marker size is fitted per figure from the measured bin pitch "
          f"(overlap x{BIN_OVERLAP})")

    # How many image pixels one 4 um bin gets inside the rasterised SVG. Below
    # about 2 the bin edges start to alias and small features lose their shape,
    # so this is the number to look at before lowering RASTER_DPI to shrink
    # files. It is measured from the real Axes rectangle, not assumed.
    if global_span and "svg" in EXPORT_FORMATS and RASTERIZE_DOTS:
        _f, _a = new_fig()
        _pos = _a.get_position()
        plt.close(_f)
        ax_w_in = _pos.width * CANVAS_W
        ax_h_in = _pos.height * CANVAS_H
        in_per_um = min(ax_w_in / global_span[0], ax_h_in / global_span[1])
        px_per_bin = RASTER_DPI * in_per_um * HALO_BIN_UM
        note = ("fine" if px_per_bin >= 2 else
                "LOW - bin edges will alias, consider a higher RASTER_DPI")
        print(f"  raster resolution: {px_per_bin:.1f} px per {HALO_BIN_UM:.0f} "
              f"um bin at RASTER_DPI={RASTER_DPI}  [{note}]")
        print(f"    halving RASTER_DPI quarters the embedded image size "
              f"and gives {px_per_bin / 2:.1f} px per bin")
    print()

    # -------------------------------------------------------------------------
    # Value layers: attach stiffness and gene expression, set colour scales
    # -------------------------------------------------------------------------
    # (column, filename suffix, cmap, vmin, vmax, bar label, kind)
    # kind is "stiffness" or "gene" and is the only thing that decides which of
    # the two value subfolders the layer is written to.
    value_specs = []

    if PLOT_STIFFNESS:
        print("Stiffness layers:")
        df, stiff_cols = attach_stiffness(df)
        for col in stiff_cols:
            v = pd.to_numeric(df[col], errors="coerce")
            label, suffix = col, col
            if col in STIFFNESS_LOG:
                with np.errstate(invalid="ignore", divide="ignore"):
                    v = np.log10(v.where(v > 0))
                label = f"log10 {col}"
                suffix = f"log10{col}"
            df["_val_" + suffix] = v
            if v.notna().sum() == 0:
                print(f"  {col}: no usable values, skipped")
                continue
            vmin, vmax = value_limits(v)
            value_specs.append(("_val_" + suffix, suffix,
                                make_cmap(STIFFNESS_CMAP, suffix),
                                vmin, vmax, label, "stiffness"))
            print(f"  {label}: colour scale {vmin:.4g} to {vmax:.4g} "
                  f"({VALUE_CLIP_PCT[0]}-{VALUE_CLIP_PCT[1]} percentile, "
                  f"scope = {VALUE_SCALE_SCOPE})")
        print()

    if GENES_TO_PLOT and WARN_ON_EXISTING_GENE_FILES:
        groot = OUTPUT_DIR / GENES_ROOT_DIR
        # Folders this run will write into, under the CURRENT grouping. Any
        # other folder in the tree is either a stale selection or the leftover
        # of a different GENE_GROUP_BY, and both mislead in the same way.
        if GENE_GROUP_BY == "tissue":
            expect = {_safe_name(t) for t in tissue_frames}
            what = "tissue"
        elif GENE_GROUP_BY == "gene":
            expect = {_safe_name(g) for g in GENES_TO_PLOT}
            what = "gene"
        else:
            expect, what = set(), "flat"
        print(f"\n[GENE FOLDERS] {groot}")
        print(f"  grouping: GENE_GROUP_BY = '{GENE_GROUP_BY}'"
              + (f", so one folder per {what}" if expect else
                 ", so everything goes in the root"))
        if not groot.exists():
            print("  does not exist yet; it will be created")
        else:
            existing = sorted(p for p in groot.iterdir() if p.is_dir())
            n_files = sum(1 for _ in groot.rglob("*") if _.is_file())
            print(f"  {len(existing)} subfolder(s) already there, "
                  f"{n_files:,} file(s) in total")
            stale = [p.name for p in existing if p.name not in expect]
            if stale:
                print(f"  {len(stale)} of them will NOT be written to by this "
                      f"run, so they are left over from an earlier gene "
                      f"selection or a different GENE_GROUP_BY:")
                print(f"    {stale[:20]}"
                      f"{' ...' if len(stale) > 20 else ''}")
                print(f"  Delete the {groot.name} folder before running if you "
                      f"want a clean set.")
            elif existing:
                print("  all of them match this run; their files will be "
                      "overwritten where names collide")

    if GENES_TO_PLOT or HANDPICKED_GENES:
        # The DEG list and the hand-picked list are loaded together, in ONE
        # pass. Two separate calls re-read the whole expression matrix and
        # re-join a multi-million-row frame for the second one, and worse, any
        # gene on both lists collided on the merge: pandas renamed it to
        # <gene>_x / <gene>_y and the plain name disappeared.
        want = list(dict.fromkeys(list(GENES_TO_PLOT) + list(HANDPICKED_GENES)))
        both = sorted(set(GENES_TO_PLOT) & set(HANDPICKED_GENES))
        print(f"Gene expression layers: {len(GENES_TO_PLOT)} from the DEG "
              f"selection + {len(HANDPICKED_GENES)} hand-picked "
              f"= {len(want)} distinct")
        if both:
            print(f"  on BOTH lists, loaded once and drawn in both places: "
                  f"{both}")
        df, all_found = attach_gene_expression(df, want)
        found = set(all_found)

        stops = ([GENE_CMAP_LO, GENE_CMAP_HI]
                 if (GENE_CMAP_LO and GENE_CMAP_HI) else GENE_CMAP_STOPS)
        gene_cmap = make_cmap(stops, "gene")
        print(f"  colour ramp: {' -> '.join(stops)}")

        # Every derived column is built into a dict and attached in a single
        # concat. One-at-a-time assignment on a frame this size is what raised
        # the "DataFrame is highly fragmented" warning.
        new_cols, pending = {}, []
        for kind, genes, prefix in (
                ("gene", [g for g in GENES_TO_PLOT if g in found], "_val_"),
                ("handpicked",
                 [g for g in HANDPICKED_GENES if g in found], "_hp_")):
            if not genes:
                continue
            if kind == "handpicked":
                print("\nHand-picked gene layers:")
                print(f"  tissues     : {HANDPICKED_TISSUES or 'all'}")
                print(f"  compartments: "
                      f"{HANDPICKED_COMPARTMENTS or 'all present'} "
                      f"(GENE_COMPARTMENTS is deliberately ignored here)")
                print(f"  folder      : {OUTPUT_DIR / HANDPICKED_DIR}")
            for g in genes:
                v = pd.to_numeric(df[g], errors="coerce")
                if GENE_ZERO_AS_BLANK:
                    v = v.where(v > 0)
                # The transformed values get their own column and THAT is
                # plotted, exactly as the stiffness branch does. Plotting the
                # raw column would blank the zeros for the colour scale only
                # and still draw them. The two prefixes keep a gene on both
                # lists from having one set overwrite the other.
                col = prefix + g
                new_cols[col] = v
                if v.notna().sum() == 0:
                    print(f"  {g}: no usable values, skipped")
                    continue
                vmin, vmax = value_limits(v)
                pending.append((col, g, gene_cmap, vmin, vmax,
                                f"{g}, log-normalised", kind))
                print(f"  {g}: colour scale {vmin:.4g} to {vmax:.4g}")

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)],
                           axis=1).copy()
        value_specs.extend(pending)

        missing = [g for g in want if g not in found]
        if missing:
            print(f"\n  [NOTE] not in the expression matrix, so not plotted "
                  f"({len(missing)}): {missing}")
        print()

    if VALUE_SCALE_SCOPE == "global":
        for _, suffix, cmap, vmin, vmax, label, _kind in value_specs:
            # A gene in both lists has two independent colour scales, so the
            # bars must not share a filename or one would silently describe
            # the other's layers.
            stem = (f"colourbar_handpicked_{suffix}"
                    if _kind == "handpicked" else f"colourbar_{suffix}")
            save_colourbar(cmap, vmin, vmax, label, stem)

    # -------------------------------------------------------------------------
    # Per-tissue plots (grouped by OvaryID)
    # -------------------------------------------------------------------------
    # Built as a job list rather than executed inline, so the same list can be
    # handed to one process or to sixty. Everything that has to happen once,
    # and in order, stays here in the parent: the colour bars, and the printed
    # per-tissue header.
    scratch_root = Path(FRAME_SCRATCH_DIR) if FRAME_SCRATCH_DIR \
        else Path(tempfile.gettempdir())
    scratch_root.mkdir(parents=True, exist_ok=True)
    if CLEAN_STALE_SCRATCH:
        sweep_stale_scratch([scratch_root, OUTPUT_DIR])
    frame_dir = Path(tempfile.mkdtemp(prefix=SCRATCH_PREFIX,
                                      dir=str(scratch_root)))
    print(f"\nWorker scratch: {frame_dir}")
    jobs = []
    frame_bytes = []

    def add(kind, frame_key, label, **args):
        jobs.append({"kind": kind, "frame": frame_key,
                     "label": label, "args": args})

    try:
        for tissue_id, (_first_pass, roi_slices) in sorted(
                tissue_frames.items()):
            # Rebuilt from the merged df by the SAME filters as the first pass,
            # rather than by index. attach_stiffness / attach_gene_expression go
            # through pd.merge, which returns a fresh index, so carrying an
            # index across would be a silent misalignment waiting to happen.
            tissue_df = df[df[COL_OVARY_ID].astype(str) == tissue_id]
            if roi_slices is not None:
                tissue_df = tissue_df[
                    tissue_df[COL_TISSUE_SLICE].astype(str).isin(roi_slices)]
            tissue_df = tissue_df.copy()

            comps_present = {parse_seg(v)[1]
                             for v in tissue_df[COL_SEG].unique()
                             if parse_seg(v)[1] is not None}
            if not comps_present:
                print(f"  {tissue_id}: no assigned ROI-Segmentation — skipping.")
                continue

            xlim, ylim = _tissue_extent(tissue_df, span=global_span)
            s = s_global if s_global is not None else marker_s(
                xlim[1] - xlim[0], ylim[1] - ylim[0])

            print(f"  {tissue_id}  [{len(tissue_df):,} bins]  "
                  f"compartments: {sorted(comps_present)}  "
                  f"pitch={measure_bin_pitch(tissue_df[COL_X].values, tissue_df[COL_Y].values):.2f} um")

            key = f"tissue_{tissue_id}"
            slim_cols = [c for c in (COL_X, COL_Y, COL_SEG, COL_COMPARTMENT)
                         if c in tissue_df.columns]
            tissue_df[slim_cols].to_pickle(frame_dir / f"{key}.pkl")
            frame_bytes.append(
                int(tissue_df[slim_cols].memory_usage(deep=True).sum()))
            base = dict(tid=tissue_id, xlim=xlim, ylim=ylim, s=s)

            for comp in ("F", "CL", "S", "V"):
                if comp in comps_present:
                    add("single_comp", key, f"{tissue_id}-{comp}",
                        comp=comp, **base)

            add("combined", key, f"{tissue_id}-ROI-Segmentation", **base)
            if PLOT_FLAT_ROI:
                add("flat", key, f"{tissue_id}{FLAT_ROI_SUFFIX}", **base)

            for comp in ("F", "CL", "S", "V"):
                if comp in comps_present:
                    add("comp_only", key, f"{tissue_id}_{comp}only",
                        comp=comp, **base)
            for comp in ("F", "CL"):
                if comp in comps_present:
                    add("features", key, f"{tissue_id}-{comp}<n>",
                        comp=comp, **base)

            for col, suffix, cmap, vmin, vmax, label, vkind in value_specs:
                if (vkind == "handpicked" and HANDPICKED_TISSUES
                        and tissue_id not in HANDPICKED_TISSUES):
                    continue
                lo, hi = vmin, vmax
                if VALUE_SCALE_SCOPE == "tissue":
                    lo, hi = value_limits(tissue_df[col])
                    if lo is None:
                        print(f"    {tissue_id}: no {suffix} values, skipped")
                        continue
                    # Stays in the parent: writing the same colour bar from
                    # several workers at once would be a torn-file race.
                    save_colourbar(cmap, lo, hi, label,
                                   f"colourbar_{tissue_id}_{suffix}")
                vkey = f"val_{tissue_id}_{suffix}"
                vcols = slim_cols + ([col] if col not in slim_cols else [])
                vdf = tissue_df[vcols]
                vdf.to_pickle(frame_dir / f"{vkey}.pkl")
                frame_bytes.append(int(vdf.memory_usage(deep=True).sum()))
                add("values", vkey, f"{tissue_id}_*_{suffix}",
                    col=col, suffix=suffix, cmap=cmap, lo=lo, hi=hi,
                    comps=comps_present, vkind=vkind, **base)

        # ---------------------------------------------------------------------
        # Per-tissue-slice compartment maps (grouped by Tissue-slices)
        # ---------------------------------------------------------------------
        print("\nPer-tissue-slice Compartments maps:")
        for slice_id in sorted(
                df[COL_TISSUE_SLICE].dropna().astype(str).unique()):
            slice_df = df[df[COL_TISSUE_SLICE].astype(str) == slice_id].copy()
            print(f"  {slice_id}  [{len(slice_df):,} bins]  "
                  f"pitch={measure_bin_pitch(slice_df[COL_X].values, slice_df[COL_Y].values):.2f} um")
            skey = f"slice_{slice_id}"
            sl_cols = [c for c in (COL_X, COL_Y, COL_COMPARTMENT)
                       if c in slice_df.columns]
            slice_df[sl_cols].to_pickle(frame_dir / f"{skey}.pkl")
            frame_bytes.append(
                int(slice_df[sl_cols].memory_usage(deep=True).sum()))
            add("slice", skey, f"{slice_id}-Compartments", sid=slice_id,
                span=global_span)

        # ---------------------------------------------------------------------
        # Draw everything
        # ---------------------------------------------------------------------
        print(f"\nRendering {len(jobs)} job(s)...")
        execute_jobs(jobs, frame_dir,
                     frame_bytes=max(frame_bytes)
                     if frame_bytes else 0)
    finally:
        # The frames are a few hundred MB of scratch; do not leave them behind
        # even if a job raised.
        remove_scratch(frame_dir)

    # -------------------------------------------------------------------------
    # Full all-CA combined map
    # -------------------------------------------------------------------------
    # One figure spanning every tissue, so there is nothing to parallelise and
    # it needs the whole frame. Drawn here in the parent.
    print("\nAll-capture-areas combined map...")
    plot_all_capture_areas(df)

    print(f"\nDone.  All files in:\n  {OUTPUT_DIR}")


def main():
    """Set up the run log, then run. Everything printed lands in both places."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_DIR / f"run_log_{stamp}.txt"

    orig_out, orig_err = _sys.stdout, _sys.stderr
    with open(log_path, "w", encoding="utf-8") as fh:
        _sys.stdout = Tee(orig_out, fh)
        _sys.stderr = Tee(orig_err, fh)
        try:
            _run()
        except Exception:
            import traceback
            traceback.print_exc()          # the traceback goes into the log too
            raise
        finally:
            print(f"\nRun log saved to: {log_path}")
            _sys.stdout, _sys.stderr = orig_out, orig_err
    print(f"Run log: {log_path}")


if __name__ == "__main__":
    main()