# Changelog

## v1.1.0

Added a demo dataset of 144 raw nanoindentation force curves from one region of
interest, with the expected results table and the console log of the reference run.

Added system requirements, installation guide, demo instructions and usage
instructions to the README.

Fixed: `np.trapz` replaced with `np.trapezoid` in `1_NI_processing_pipeline.py`
(9 occurrences). `np.trapz` was removed in NumPy 2.0, so the script would not run
on the documented environment. The two functions are identical; re-running the
pipeline on one region of interest reproduced all 144 modulus and energy values
exactly, with a maximum absolute difference of 0.0.

Added `openpyxl` to the dependency list. It is required by
`1_NI_processing_pipeline.py` to write `maps.xlsx` and was previously undocumented.

Regenerated `requirements.txt` and `conda_packages.txt`.

## v1.0.0

Initial release accompanying the manuscript submission.
