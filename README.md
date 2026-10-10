# Correlative Spatial Mechanomics

**Linking local tissue stiffness to the molecular state of the same tissue region.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22825608.svg)](https://doi.org/10.5281/zenodo.22825608)

Preprint: [https://doi.org/10.64898/2026.08.29.748050](https://doi.org/10.64898/2026.08.29.748050)

---

## The question

How does the stiffness of a small region of tissue relate to what the cells in that
region are doing?

The difficulty is that stiffness and gene expression cannot be measured on the same
piece of tissue while preserving native tissue mechanics. Measuring mechanics needs
fresh, hydrated tissue. Measuring the transcriptome needs tissue that has been frozen,
fixed and sectioned. One destroys the conditions the other requires.

## The approach

A single vibratome cut produces two facing surfaces of the same tissue.

- **Surface 1** stays fresh and is mapped by nanoindentation, giving a Young's modulus
  every 50 µm across a millimetre-scale area.
- **Surface 2**, immediately beneath and parallel to it, goes to 10x Genomics Visium HD
  spatial transcriptomics at 4 µm resolution.

The two are then brought into one coordinate frame by matching the vascular network,
which runs through the full thickness of the cut and so appears on both faces. Every
stiffness measurement ends up paired with the transcriptome of the tissue directly
beneath it.

<p align="center">
  <em>Fresh surface → stiffness map &nbsp;|&nbsp; Adjacent surface → spatial transcriptome &nbsp;|&nbsp; Aligned on shared vasculature</em>
</p>

## Applied to the ageing ovary

The ovary is a good test case: follicles, corpora lutea, stroma and vasculature sit
side by side within a millimetre, and the tissue remodels extensively with age.

|  |  |
|---|---|
| Matched stiffness–transcriptome measurements | **2,771** |
| Regions of interest | **21** |
| Ovaries / animals | **8 / 6** |
| Transcriptome bins per measurement | **~89** |

This let us ask which molecular programs track local stiffness, how much of the
variation cell composition alone can explain, and how these relationships change
with age.

---

## What is in this repository

Eighteen scripts covering the full workflow, from raw force curves to the final
correlations. Each carries its own docstring with parameters, inputs and outputs.

### Mechanics
| # | Script | Does |
|---|---|---|
| 1 | `1_NI_processing_pipeline.py` | Fits raw force–displacement curves |
| 2 | `2_NI_qc_indentation_depth.py` | Indentation depth quality control |
| 3 | `3_NI_qc_sample_summary.py` | Per-slice point counts and QC survival |
| 4 | `4_NI_parameter_summary.py` | All mechanical parameters by slice and age |
| 5 | `5_NI_stiffness_stats.py` | Mixed models, age and compartment comparisons |
| 6 | `6_NI_stiffness_map.py` | Stiffness heatmaps per region |

### Spatial transcriptomics
| # | Script | Does |
|---|---|---|
| 7 | `7_ST_Comparing2_4_8um.py` | Compares 2, 4 and 8 µm binning |
| 8 | `8_Multisample-integration-VisiumHD.Rmd` | Integration, clustering, cell annotation |

### Registration and correlation
| # | Script | Does |
|---|---|---|
| 9 | `9_Corr_roi_bin_assignment.py` | Assigns 4 µm bins to 50 µm grid points |
| 10 | `10_DEG_GSEA_YvA_wMechanics.Rmd` | Differential expression and enrichment |
| 11 | `11_curate_gsea_dotplots.py` | Curates enrichment dot plots |
| 12 | `12_Corr_stiffness_expression_correlation.py` | Stiffness–expression correlations |
| 13 | `13_merge_correlation_hits.py` | Merges results, builds summary tables |
| 14 | `14_plot_segmentation.py` | Spatial plots of compartments and genes |

### Controls
| # | Script | Does |
|---|---|---|
| 15 | `15_depth_confound_diagnostic.py` | Why correlation fails at bin level |
| 16 | `16_positive_controls.py` | Registration positive controls |

### Standalone
| # | Script | Does |
|---|---|---|
| 17 | `17_Extra_celltype_mechanics_regression.py` | Composition-to-stiffness regression |
| 18 | `18_tilt_correction.ijm` | Tilt correction of confocal stacks in Fiji |

Full details of each script, the order to run them in, what depends on what, and the
settings worth knowing about are in **[`README.txt`](README.txt)**.

---

## 1. System requirements

**Software dependencies**

Python 3.13.13 with numpy 2.4.6, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.0,
statsmodels 0.14.6, patsy 1.0.2, matplotlib 3.11.0, seaborn 0.13.2, gseapy 1.3.0
and openpyxl 3.1.5. Script 1 uses `np.trapezoid` and therefore requires NumPy 2.0
or later.

R 4.5.1 with Seurat 5.3.1, SeuratObject 5.2.0, BPCells 0.3.1, SeuratWrappers 0.4.0,
Banksy 1.6.0, harmony 1.2.4, SpatialExperiment 1.20.0, scran 1.38.1, scater 1.38.1,
clustree 0.5.1, DESeq2 1.50.2, apeglm 1.32.0, fgsea 1.36.2 and msigdbr 26.1.0.

Fiji (ImageJ 1.53q, Java 1.8.0_322) is required for script 18 only. Imaris 10.2.0
and SpatialX v2026-08-03 were used for image projection and manual alignment.

**Operating systems**

Tested on Windows Server 2019 (x86_64). The Python scripts use no OS-specific calls
and are expected to run on macOS and Linux, though this has not been tested.

**Non-standard hardware**

None for the demo, which runs on any desktop. Script 8 integrates 3.7 million
transcriptome bins across four capture areas and was run on a workstation with
512 GB RAM; at least 64 GB is advisable for that step. All other scripts run within
a few GB.

## 2. Installation guide

1. Install Python 3.13 and R 4.5.1.
2. Download or clone this repository.
3. Install the Python packages, for example
   `conda install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn openpyxl`
   followed by `pip install gseapy`.
4. Install the R packages from CRAN and Bioconductor.
5. No compilation or build step is required; the scripts run directly.

**Typical install time on a standard desktop:** approximately 2 minutes for the
Python packages and 20 to 30 minutes for the R packages, dominated by Bioconductor
compilation.

Exact versions as run are in [`requirements.txt`](requirements.txt),
[`conda_packages.txt`](conda_packages.txt) and
[`R_sessionInfo.txt`](R_sessionInfo.txt). These record what was installed rather
than providing a one-command install.

## 3. Demo

A single nanoindentation region of interest is included in `demo_data/`: 144 raw
force-displacement curves from a young mouse ovary, slice
`20240920_13w-01-ms02-young-Y1-2`, region Y1-2.

**To run:**

```
python 1_NI_processing_pipeline.py 0
```

with `demo_data/20240920_13w-01-ms02-young-Y1-2/` placed in the same folder as the
script. The trailing `0` selects the first slice folder found.

**Expected output**, written to `demo_data/20240920_13w-01-ms02-young-Y1-2/output/`:

| Output | Contents |
|---|---|
| `NI_..._all_samples.csv` | one row per curve: Young's modulus, effective and retraction moduli, viscosity, loss ratio, energies, fit errors and grid coordinates |
| `maps.xlsx`, `correlations.csv` | per-parameter maps and pairwise correlations |
| `..._maps.png` / `.pdf` / `.jpg` / `.tiff` | parameter maps across the region |
| one folder per curve, plus `viscosity/` | per-curve diagnostic plots |
| `NI_tissue_summary_stats.csv` | per-slice summary, written beside the script rather than in the output folder |

Of the 144 curves, 138 return a fitted modulus and all 138 pass the Hertzian fit
criterion (`Perr_l` ≤ 0.15). A reference copy of the results table is in
`demo_expected_output/`; the image and per-curve outputs are not included, as the
TIFF alone exceeds 70 MB. The console output of the reference run is in
[`demo_run_log.txt`](demo_run_log.txt).

**Expected run time on a standard desktop:** approximately 2 minutes for curve
fitting, and under 10 minutes in total including figure output.

## 4. Instructions for use

To run on your own data, place one folder per nanoindentation slice beneath the
script, each containing the raw `.txt` force curves, and name each folder with a
leading eight-digit date. Edit the paths and acquisition parameters at the top of
each script, in particular the probe radius `rb` and Poisson's ratio `mu`, then run
the scripts in the order below.

```
1 ──▶ 2, 3, 4, 5, 6
8 ──▶ 10
9 ──▶ 12, 14, 15, 16, 17
10 ──▶ 5, 11, 12, 13, 14, 15, 16
12 ──▶ 13
```

Scripts 17 and 18 stand alone.

Folder layout and metadata column names are assumed throughout; renaming either
requires updating the script that reads them. Four scripts need a decision or a
manual edit partway through: 5, 9, 11 and 18. See [`README.txt`](README.txt) for the
full manifest, dependency graph and the settings worth knowing about.

Random seeds are fixed throughout.

---

## Data

Spatial transcriptomics data are deposited in GEO under accession GSE346700,
currently private pending publication. Mechanical data, registration images and
derived tables are available as described in the associated publication.

## Citing this code

This release is archived with a DOI. Please cite the DOI rather than the repository
URL, so the citation points at this exact version of the code.

## Licence

MIT. See [`LICENSE`](LICENSE).

---

<sub>Bringing mechanobiology and spatial biology together, one tissue surface at a time.</sub>
