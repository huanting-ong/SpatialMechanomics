"""
3_NI_qc_sample_summary.py

Counts how many indentation points each slice produced and how many survived
each stage of quality control.

Parameters
    Perr_l <= 0.15  Hertzian fit relative standard error, used for the
                    successful_Elasticity_l_nonNaN_HertzFit column. Matches
                    script 1 and the analysis reported in the manuscript.
                    Deliberately looser than the conventional 0.05, which
                    scales inversely with modulus and would discard soft
                    tissue preferentially

Actions required
    Place this script in the project root, with one folder per slice directly
    beneath it. Run script 1 first, since this reads its output.
        py 3_NI_qc_sample_summary.py
    A slice with no all samples.csv is reported as MISSING rather than skipped,
    so an unprocessed slice cannot disappear from the count silently.

Input
    <root>/<slice>/output/all samples.csv     from script 1
Output
    <root>/<slice>/output/summary_counts.csv  one row for that slice
    <root>/all_summary_counts.csv             every slice, including MISSING
Columns
    unique_X, unique_Y, grid_X_times_Y        grid dimensions and their product
    actual_points_in_csv                      rows present
    successful_indentation                    curves where contact was detected
    successful_Elasticity_l_nonNaN            curves that returned a modulus
    successful_Elasticity_l_nonNaN_HertzFit   the subset also passing Perr_l
"""

import os
import pandas as pd
import numpy as np
import sys

root = os.path.dirname(__file__)+'/' # the filefolder which this py file is located

def summarise_csv(csv_path, slice_name):
    df = pd.read_csv(csv_path)

    unique_X = df["X"].nunique() if "X" in df.columns else 0
    unique_Y = df["Y"].nunique() if "Y" in df.columns else 0
    grid_X_times_Y = unique_X * unique_Y
    actual_points = len(df)

    if "IndentationPresent" in df.columns:
        successful_indentation = int(df["IndentationPresent"].sum())
    else:
        successful_indentation = 0

    if "Elasticity_l (Pa)" in df.columns:
        successful_elasticity = int(df["Elasticity_l (Pa)"].notna().sum())
        if "Perr_l" in df.columns:
            successful_elasticity_hf = int(
                ((df["Elasticity_l (Pa)"].notna()) & (df["Perr_l"] <= 0.15)).sum()
            )
        else:
            successful_elasticity_hf = np.nan
    else:
        successful_elasticity = 0
        successful_elasticity_hf = np.nan

    return {
        "Slice": slice_name,
        "CSV": csv_path,
        "status": "OK",
        "unique_X": unique_X,
        "unique_Y": unique_Y,
        "grid_X_times_Y": grid_X_times_Y,
        "actual_points_in_csv": actual_points,
        "successful_indentation": successful_indentation,
        "successful_Elasticity_l_nonNaN": successful_elasticity,
        "successful_Elasticity_l_nonNaN_HertzFit": successful_elasticity_hf,
    }

def missing_row(slice_name):
    return {
        "Slice": slice_name,
        "CSV": os.path.join(root, slice_name, "output", "all samples.csv"),
        "status": "MISSING",
        "unique_X": np.nan,
        "unique_Y": np.nan,
        "grid_X_times_Y": np.nan,
        "actual_points_in_csv": np.nan,
        "successful_indentation": np.nan,
        "successful_Elasticity_l_nonNaN": np.nan,
    }

def main():
    combined = []

    for slice_name in os.listdir(root):
        slice_path = os.path.join(root, slice_name)
        if not os.path.isdir(slice_path):
            continue

        csv_path = os.path.join(slice_path, "output", "all samples.csv")
        if os.path.exists(csv_path):
            print(f"[OK] Found: {csv_path}")
            summary = summarise_csv(csv_path, slice_name)
            # per-slice file only for OK
            out_csv = os.path.join(slice_path, "output", "summary_counts.csv")
            pd.DataFrame([summary]).to_csv(out_csv, index=False)
            print(f"    → Wrote {out_csv}")
            combined.append(summary)
        else:
            print(f"[MISS] No all samples.csv in {slice_name}")
            combined.append(missing_row(slice_name))

    if combined:
        combined_csv = os.path.join(root, "all_summary_counts.csv")
        pd.DataFrame(combined).to_csv(combined_csv, index=False)
        print(f"\n[OK] Wrote combined summary (with OK & MISSING): {combined_csv}")
    else:
        print("\n[FATAL] No slice folders found.")
        sys.exit(0)

if __name__ == "__main__":
    main()
