CORRELATIVE SPATIAL MECHANOMICS
Analysis code manifest

Paired nanoindentation and Visium HD spatial transcriptomics on opposing faces
of a single vibratome cut. Eighteen scripts, each one carries its own docstring 
with parameters, inputs and outputs; this file is the map between them.


-------------------------------------------------------------------------
SCRIPTS
-------------------------------------------------------------------------
  #   File                                        Language
  --  ------------------------------------------  --------
   1  1_NI_processing_pipeline.py                  Python
   2  2_NI_qc_indentation_depth.py                 Python
   3  3_NI_qc_sample_summary.py                    Python
   4  4_NI_parameter_summary.py                    Python
   5  5_NI_stiffness_stats.py                      Python
   6  6_NI_stiffness_map.py                        Python
   7  7_ST_Comparing2_4_8um.py                     Python
   8  8_Multisample-integration-VisiumHD.Rmd       R
   9  9_Corr_roi_bin_assignment.py                 Python
  10  10_DEG_GSEA_YvA_wMechanics.Rmd               R
  11  11_curate_gsea_dotplots.py                   Python
  12  12_Corr_stiffness_expression_correlation.py  Python
  13  13_merge_correlation_hits.py                 Python
  14  14_plot_segmentation.py                      Python
  15  15_depth_confound_diagnostic.py              Python
  16  16_positive_controls.py                      Python
  17  17_Extra_celltype_mechanics_regression.py    Python
  18  18_tilt_correction.ijm                       ImageJ macro


-------------------------------------------------------------------------
WHAT EACH ONE DOES
-------------------------------------------------------------------------
MECHANICS  (run from the nanoindentation project root, one folder per slice
            directly beneath it)

   1  Fits raw force-displacement .txt files. Returns Young's modulus, the
      effective and retraction moduli, viscosity, loss ratio and energies.
      Writes one result per point and one all-samples CSV per slice.
   2  Indentation depth QC. Flags curves indented beyond the Hertz fit range.
      Reads the raw curves, so it can run before or after script 1.
   3  Per-slice point counts and survival through each QC criterion.
   4  Every mechanical parameter summarised by slice and by age group.
   5  Stiffness statistics. Linear mixed models with animal, ovary and ROI as
      nested random effects. Produces the age comparison, the compartment
      contrasts, and the Prism-ready exports.
   6  Stiffness heatmaps per ROI on a shared colour scale.

SPATIAL TRANSCRIPTOMICS  (run from the Visium HD project folder)

   7  Compares 2, 4 and 8 um binning. Justifies the 4 um choice.
   8  Integrates the four capture areas. BANKSY neighbourhood augmentation,
      Harmony batch correction, clustering, marker genes, manual annotation of
      the 38 populations, and the mapping of those populations to the four
      compartments.

REGISTRATION AND CORRELATION  (run from the SpatialX analysis folder)

   9  Recovers the nanoindentation grid from corner annotations and assigns
      4 um transcriptome bins to each 50 um grid point. Writes
      metadata_with_nano.tsv, the input to 12, 14, 15, 16 and 17.
  10  Pseudobulk differential expression and gene set enrichment, aged versus
      young, whole tissue and per compartment. Also exports the expression
      matrix and the matched metadata table that scripts 5, 12, 15 and 16
      read.
  11  Curates which enrichment terms appear on a dot plot. Two passes,
      template then replot, with a manual edit between them.
  12  Partial Spearman correlation between per-grid-point expression and
      stiffness, pooled and stratified by age, whole tissue and per
      compartment.
  13  Merges the correlation results across analyses and builds the Ovarian
      Mechanomics summary table.
  14  Spatial plots of compartments, stiffness and selected genes.

CONTROLS

  15  Demonstrates why correlation fails at 4 um bin level and why script 12
      aggregates to the 50 um grid point first. Grid points are keyed on
      Tissue.slices + ROIs + nano_point_id, since nano_point_id alone is a
      grid index that repeats across ROIs and tissues. Counts are summed and
      normalised once per grid point, matching script 12.
  16  Registration positive controls: spatial autocorrelation of stiffness,
      stiffness by compartment, composition against stiffness with ovary
      identity held constant, and an ovary-level association reported as a
      scale rather than as a test. Compartments are assigned by plurality
      here, so no measurement is discarded; the reported compartment analyses
      require 50%, and the run log prints both counts.

STANDALONE

  17  Elastic net regression predicting stiffness from cell type composition.
      Found little here, since composition barely predicts stiffness, and no
      output appears in the manuscript. Provided for use on other tissues.
  18  Tilt correction for post-nanoindentation confocal stacks, run in Fiji.
      Reslices from the left, applies a per-stack rotation measured with the
      line tool, and reslices from the top, so the tissue surface lies
      parallel to the imaging plane and the 12 um projection used for image
      alignment is measured perpendicular to the surface. Image alignment
      itself is performed manually and is not scripted.


-------------------------------------------------------------------------
WHAT DEPENDS ON WHAT
-------------------------------------------------------------------------
   1  -->  2, 3, 4, 5, 6
   8  -->  10
   9  -->  12, 14, 15, 16, 17
  10  -->  5, 11, 12, 13, 14, 15, 16
  12  -->  13

Scripts 17 and 18 stand alone. 18 is an image preprocessing step used before
manual alignment and feeds nothing downstream in code.

Script 10 writes the expression matrix as expression_export/step9_*.mtx and
related files. That prefix dates from when it was script 9 and is read under
that name by script 12; it is left unchanged deliberately.

Scripts 5, 12, 15 and 16 all read step9_expression_metadata_matched.tsv, so
that every analysis uses the same set of expression-matched bins and the
compartment counts agree between them.


-------------------------------------------------------------------------
DEPENDENCIES
-------------------------------------------------------------------------
Python 3.13.13          numpy 2.4.6, pandas 2.3.3, scipy 1.17.1,
                        scikit-learn 1.9.0, statsmodels 0.14.6, patsy 1.0.2,
                        matplotlib 3.11.0, seaborn 0.13.2, gseapy 1.3.0,
                        openpyxl 3.1.5
R 4.5.1                 Seurat 5.3.1, SeuratObject 5.2.0, BPCells 0.3.1,
                        SeuratWrappers 0.4.0, Banksy 1.6.0, harmony 1.2.4,
                        SpatialExperiment 1.20.0, scran 1.38.1, scater 1.38.1,
                        clustree 0.5.1, DESeq2 1.50.2, apeglm 1.32.0,
                        fgsea 1.36.2, msigdbr 26.1.0
Fiji                    ImageJ 1.53q, Java 1.8.0_322, 64-bit; script 18 only
Imaris                  10.2.0 (Oxford Instruments); maximum intensity
                        projections of the tilt-corrected stacks
SpatialX (BioTuring)    v2026-08-03, manual image alignment and ROI annotation

Script 1 uses np.trapezoid and therefore requires NumPy 2.0 or later. It also
requires openpyxl, which it uses to write maps.xlsx.

Full environment listings as captured on the analysis workstation are in
requirements.txt (Python) and R_sessionInfo.txt (R). Analyses were run under
Windows Server 2019, x86_64.

requirements.txt and conda_packages.txt record the exact package versions
present on the analysis workstation, captured by pip freeze and conda list
--export respectively. Neither reconstructs the environment automatically:
conda-installed packages appear in the pip file as local build paths, and
pip-installed packages appear in the conda file as pypi_0. Install the
packages named above by name and version.
Random seeds are fixed throughout (RANDOM_STATE 42; SEED 1000 in script 8).
Parallel and serial execution paths were verified to produce identical output.


-------------------------------------------------------------------------
BEFORE YOU RUN
-------------------------------------------------------------------------
Paths are hard-coded at the top of each script. Edit them first.

Each script assumes a folder layout and specific column names. Do not rename
folders or metadata columns without updating the script that reads them.

Four scripts need a decision or an edit partway through:

   5   SAMPLE_SHEET maps each slice to an ovary and an animal. Two ovaries
       from one mouse must share an animal id, or the age test is too
       generous.
   9   Corner bins must be annotated in SpatialX under the ROI_boundary
       column before this will run.
  11   Chunk order is template, then a manual edit of the *_toplot.csv files,
       then replot.
  18   The rotation angle is measured separately for every stack with the line
       tool and edited into the macro before running. The value committed here
       is an example, not a constant.

Three settings recur and are worth knowing about:

  VALID_MIN, VALID_MAX   7.6 to 22,726 Pa. Moduli outside this are treated as
                         physically implausible for this tissue and probe and
                         set to NaN.
  Perr_l                 0.15, not the conventional 0.05. This quantity scales
                         inversely with modulus, so a strict cut discards soft
                         tissue preferentially and biases an age comparison.
  COMPARTMENT_MIN_FRAC   0.50. A grid point is assigned to a compartment when
                         at least half its expression-matched bins belong to
                         it. Points reaching 50% in nothing are used only in
                         whole-tissue analyses. Script 16 deliberately does
                         not apply this; see above.

A worked example is included. demo_data/ holds 144 raw force curves from one
region of interest, and demo_expected_output/ holds the results table script 1
produces from them. See README.md for how to run it.


-------------------------------------------------------------------------
LICENSE
-------------------------------------------------------------------------
MIT. See LICENSE.