CORRELATIVE SPATIAL MECHANOMICS
Analysis code manifest

Paired nanoindentation and Visium HD spatial transcriptomics on opposing faces
of a single vibratome cut. Seventeen scripts, run in the order below. Each one
carries its own docstring with parameters, inputs and outputs; this file is the
map between them.


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
      4 um transcriptomic bins to each 50 um measurement. Writes
      metadata_with_nano.tsv, the input to 5, 12, 14, 15, 16 and 17.
  10  Pseudobulk differential expression and gene set enrichment, aged versus
      young, whole tissue and per compartment. Also exports the expression
      matrix that script 12 reads.
  11  Curates which enrichment terms appear on a dot plot. Two passes,
      template then replot, with a manual edit between them.
  12  Partial Spearman correlation between per-measurement expression and
      stiffness, pooled and stratified by age, whole tissue and per
      compartment.
  13  Merges the correlation results across analyses and builds the Ovarian
      Mechanomics summary table.
  14  Spatial plots of compartments, stiffness and selected genes.

CONTROLS

  15  Demonstrates why correlation fails at 4 um bin level and why script 12
      aggregates to the 50 um measurement first.
  16  Registration positive controls: spatial autocorrelation of stiffness,
      stiffness by compartment, and composition against stiffness.

EXTRAS

  17  Elastic net regression predicting stiffness from cell type composition.
      Found little here, since composition barely predicts stiffness, and no
      output appears in the manuscript. Provided for use on other tissues.


-------------------------------------------------------------------------
WHAT DEPENDS ON WHAT
-------------------------------------------------------------------------
   1  -->  2, 3, 4, 5, 6
   8  -->  10
   9  -->  5, 12, 14, 15, 16, 17
  10  -->  11, 12, 13, 14, 15
  12  -->  13

Script 10 writes the expression matrix as expression_export/step9_*.mtx and
related files. That prefix dates from when it was script 9 and is read under
that name by script 12; it is left unchanged deliberately.


-------------------------------------------------------------------------
DEPENDENCIES
-------------------------------------------------------------------------
Python 3.13.13   	numpy 2.4.6, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.0,
             		statsmodels 0.14.6, patsy 1.0.2, matplotlib 3.11.0,
           		seaborn 0.13.2, gseapy 1.3.0
R v4.5.1    	       	Seurat 5.3.1, BPCells 0.3.1, SeuratWrappers 0.4.0, Banksy 1.6.0,
            		harmony 1.2.4, SpatialExperiment 1.20.0, scran 1.38.1,
             		scater 1.38.1, clustree 0.5.1, DESeq2 1.50.2, apeglm 1.32.0,
             		fgsea 1.36.2, msigdbr 26.1.0

Random seeds are fixed throughout (RANDOM_STATE 42; SEED 1000 in script 8).
Parallel and serial execution paths were verified to produce identical output.


-------------------------------------------------------------------------
BEFORE YOU RUN
-------------------------------------------------------------------------
Paths are hard-coded at the top of each script. Edit them first.

Each script assumes a folder layout and specific column names. Do not rename
folders or metadata columns without updating the script that reads them.

Three scripts need a decision or an edit partway through:

   5   SAMPLE_SHEET maps each slice to an ovary and an animal. Two ovaries
       from one mouse must share an animal id, or the age test is too
       generous.
   9   Corner bins must be annotated in SpatialX under the ROI_boundary
       column before this will run.
  11   Chunk order is template, then a manual edit of the *_toplot.csv files,
       then replot.

Two settings recur and are worth knowing about:

  VALID_MIN, VALID_MAX   7.6 to 22,726 Pa. Moduli outside this are treated as
                         physically implausible for this tissue and probe and
                         set to NaN.
  Perr_l                 0.15, not the conventional 0.05. This quantity scales
                         inversely with modulus, so a strict cut discards soft
                         tissue preferentially and biases an age comparison.