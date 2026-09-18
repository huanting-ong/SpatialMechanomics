# Correlative Spatial Mechanomics

**Linking local tissue stiffness to the molecular state of the same tissue region.**

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

## Running it

```
1 ──▶ 2, 3, 4, 5, 6
8 ──▶ 10
9 ──▶ 12, 14, 15, 16, 17
10 ──▶ 5, 11, 12, 13, 14, 15, 16
12 ──▶ 13
```

Scripts 17 and 18 stand alone.

**Paths are hard-coded at the top of each script. Edit them first.**

Four scripts need a decision or a manual edit partway through: 5, 9, 11 and 18. See
`README.txt`.

## Requirements

**Python 3.13.13** with numpy, pandas, scipy, scikit-learn, statsmodels, patsy,
matplotlib, seaborn and gseapy.

**R 4.5.1** with Seurat, BPCells, SeuratWrappers, Banksy, harmony,
SpatialExperiment, scran, scater, clustree, DESeq2, apeglm, fgsea and msigdbr.

**Also used:** Fiji (ImageJ 1.53q), Imaris 10.2.0, SpatialX v2026-08-03.

Exact versions as run are in [`requirements.txt`](requirements.txt),
[`conda_packages.txt`](conda_packages.txt) and
[`R_sessionInfo.txt`](R_sessionInfo.txt). These record what was installed rather than
providing a one-command install; install the packages above by name and version.

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
